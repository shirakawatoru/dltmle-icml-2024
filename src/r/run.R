library(argparse)
library(ltmle)

main <- function() {
    # Create an argument parser
    parser <- ArgumentParser(description = "Run experiments with ltmle")
    parser$add_argument("--seed", type = "integer", default = 1234, help = "A random seed")
    parser$add_argument("--n_sim", type = "integer", default = 100, help = "A number of simulations")
    parser$add_argument("--artifact_path", type = "character", default = "artifact", help = "path to artifact directory")
    parser$add_argument("--data_name", type = "character", default = "simple-n1000-t10", help = "simple-n1000-t10")
    parser$add_argument("--model_name", type = "character", default = "glm", help = "glm;sl")
    parser$add_argument("--continuous_outcome", action = "store_true", help = "Set to TRUE if continuous outcome is needed")
    parser$add_argument("--overwrite", action = "store_true", help = "Set to TRUE if overwrite is needed")
    parser$add_argument("--verbose", action = "store_true", help = "Set to TRUE if verbose output is needed")
    parser$add_argument("--python_command", type = "character", default = "python", help = "Path to Python executable")
    args <- parser$parse_args()

    print(paste("n_sim:", args$n_sim))
    print(paste("data_name:", args$data_name))
    print(paste("model_name:", args$model_name))
    print(paste("overwrite:", args$overwrite))

    run(args)
}

run <- function(args) {
    ltmle_args <- get_ltmle_args(args$model_name)
    fit_and_save(args, ltmle_args)

    summarize(args)
}

get_ltmle_args <- function(model_name) {
    if (model_name == "glm") {
        list(
            SL.library = "glm",
            SL.cvControl = list(),
            variance.method = "ic"
        )
    } else if (model_name == "sl") {
        list(
            SL.library = c("SL.glm", "SL.earth", "SL.xgboost"),
            SL.cvControl = list(V = 3),
            variance.method = "ic"
        )
    } else {
        stop("Unknown model_name")
    }
}

fit_and_save <- function(args, ltmle_args) {
    for (b in seq(args$n_sim)) {
        fit_file_path <- get_fit_file_path(b, args)
        if (!args$overwrite && file.exists(fit_file_path)) {
            print(paste("Skip", fit_file_path))
            next
        }

        start_time <- Sys.time()

        .fit <- fit_single_ltmle(b, args, ltmle_args)

        end_time <- Sys.time()

        # Compute the elapsed time in seconds
        elapsed_time <- as.numeric(difftime(end_time, start_time, units = "secs"))

        if (args$verbose) {
            print(b)
            print(summary(.fit))
            print(paste("Elapsed time:", elapsed_time))
        }

        .fit$elapsed_time <- elapsed_time

        get_fit_dir_path(args, creatr_if_not_exists = TRUE)

        saveRDS(.fit, file = fit_file_path)
        rm(.fit)
    }
}

fit_single_ltmle <- function(b, args, ltmle_args) {
    data <- gen_data(b, args)

    c_nodes <- grep("^C_", names(data$data))
    for (j in c_nodes) {
        data$data[, j] <- BinaryToCensoring(is.censored = data$data[, j])
    }

    fit <- ltmle(
        data = data$data,
        Anodes = grep("^A_", names(data$data)),
        Lnodes = grep("^[WL]_", names(data$data)),
        Cnodes = c_nodes,
        Ynodes = grep("^Y_", names(data$data)),
        survivalOutcome = !args$continuous_outcome,
        SL.library = ltmle_args$SL.library,
        SL.cvControl = ltmle_args$SL.cvControl,
        variance.method = ltmle_args$variance.method,
        abar = data$abar1
    )

    list(
        estimates = fit$estimates,
        IC = fit$IC
    )
}

gen_data <- function(b, args) {
    data_path <- get_data_dir_path(args)

    command <- paste(
        args$python_command,
        "src/r/gen_data.py",
        "--seed", args$seed,
        "-b", b - 1,
        "--data_name", args$data_name,
        "--output_path", data_path
    )
    if (args$verbose) {
        print(command)
    }
    system(command)

    list(
        data = read.csv(file.path(data_path, "ltmle.csv")),
        abar0 = as.matrix(read.csv(file.path(data_path, "abar0.csv"))),
        abar1 = as.matrix(read.csv(file.path(data_path, "abar1.csv")))
    )
}

get_data_dir_path <- function(args) {
    file.path(args$artifact_path, "ltmle", "data", args$data_name, args$model_name)
}

get_fit_dir_path <- function(args, creatr_if_not_exists = FALSE) {
    path <- file.path(args$artifact_path, "ltmle", "fit", args$data_name, args$model_name)
    if (creatr_if_not_exists && !dir.exists(path)) {
        dir.create(path, recursive = TRUE)
    }
    path
}

get_fit_file_path <- function(b, args) {
    fit_data_name <- sprintf("b%03d_fit.rds", b - 1)
    file.path(get_fit_dir_path(args), fit_data_name)
}

get_fit_data_name <- function(b) {
    sprintf("b%03d_fit.rds", b - 1)
}

load_fits <- function(args) {
    lapply(
        seq(args$n_sim),
        function(b) {
            print(paste("Loading", get_fit_file_path(b, args)))
            readRDS(file = get_fit_file_path(b, args))
        }
    )
}

summarize <- function(args) {
    fits <- load_fits(args)
    n <- nrow(gen_data(1, args)$data)

    df <- data.frame(
        est = vapply(fits, function(fit) {
            fit$estimates[1]
        }, 0.0),
        se = vapply(fits, function(fit) {
            sqrt(mean(fit$IC$tmle**2) / n)
        }, 0.0),
        PnIC = vapply(fits, function(fit) {
            mean(fit$IC$tmle)
        }, 0.0),
        time = vapply(fits, function(fit) {
            mean(fit$elapsed_time)
        }, 0.0)
    )

    out_dir <- file.path("results", "eval", args$data_name, "ltmle")
    if (!dir.exists(out_dir)) {
        dir.create(out_dir, recursive = TRUE)
    }

    write.csv(
        df,
        file.path(out_dir, sprintf("%s.csv", args$model_name)),
        row.names = FALSE
    )
}

main()
