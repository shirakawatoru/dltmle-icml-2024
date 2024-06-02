import argparse

from src.exec.tune import tune
from src.exec.run import run
from src.exec.summarize import summarize
from src.exec.summarize_fig1 import summarize_fig1
from src.exec.gen_data import gen_data
from src.exec.make_table import make_table
from src.exec.estimate import estimate
from src.exec.plot import plot

def main():
    parser = argparse.ArgumentParser(description='CLI for deep ltmle research')
    
    subparsers = parser.add_subparsers(title='commands', dest='command')
    subparsers.required = True

    # tune command
    parser_tune = subparsers.add_parser('tune', help='tune hyperparameter')
    parser_tune.add_argument("--name", default="default_run")
    parser_tune.add_argument("--seed", type=int, default=1234)
    parser_tune.add_argument("--n_random_search", type=int, default=50)
    parser_tune.add_argument("--data_name", type=str, default="simple-n1000-t10", help="simple-n1000-t10;complex-n1000-t10-p5-h5")
    parser_tune.add_argument("--configuration_name", type=str, default="dltmle", help="dltmle")
    parser_tune.add_argument("--artifact_path", type=str, default="artifact/")
    parser_tune.add_argument("--use_cpu", action="store_true")
    parser_tune.add_argument("--overwrite", action="store_true")
    parser_tune.set_defaults(func=tune)

    # run command
    parser_run = subparsers.add_parser('run', help='run experiment')
    parser_run.add_argument("--name", default="default_run")
    parser_run.add_argument("--seed", type=int, default=1234)
    parser_run.add_argument("--data_name", type=str, default="simple-n1000-t10", help="simple-n1000-t10;complex-n1000-t10-p5-h5")
    parser_run.add_argument("--configuration_name", type=str, default="dltmle", help="dltmle;ltmle;deepace")
    parser_run.add_argument("--n_sim", type=int, default=10)
    parser_run.add_argument("--artifact_path", type=str, default="artifact/")
    parser_run.add_argument("--use_cpu", action="store_true")
    parser_run.add_argument("--overwrite", action="store_true")
    parser_run.add_argument("--overwrite_model", action="store_true")
    parser_run.add_argument("--hparams", type=str, default=None)
    parser_run.add_argument("--checkpoint", type=str, default=None)
    parser_run.set_defaults(func=run)

    # summarize command
    parser_summarize = subparsers.add_parser('summarize', help='make result summary')
    parser_summarize.add_argument("--name", default="default_run")
    parser_summarize.add_argument("--data_name", type=str, default="simple-n1000-t10", help="simple-n1000-t10;complex-n1000-t10-p5-h5")
    parser_summarize.add_argument("--exclude_from_plot", type=str, nargs="+", help="exclude from plot")
    parser_summarize.set_defaults(func=summarize)
    
    # summarize_fig1 command
    parser_summarize_fig1 = subparsers.add_parser('summarize_fig1', help='make result summary')
    parser_summarize_fig1.add_argument("--data_name", type=str, default="lay-cont-t10", help="lay-cont-t10")
    parser_summarize_fig1.add_argument("--exclude_from_plot", type=str, nargs="+", default=["dltmle\u2020"], help="exclude from plot")
    parser_summarize_fig1.set_defaults(func=summarize_fig1)

    # gen_data command
    parser_gen_data = subparsers.add_parser('gen_data', help='make result summary')
    parser_gen_data.add_argument("--name", default="default_run")
    parser_gen_data.add_argument("--seed", type=int, default=1234)
    parser_gen_data.add_argument("--n_dataset", type=int, default=500)
    parser_gen_data.add_argument("--artifact_path", type=str, default="artifact/")
    parser_gen_data.add_argument("--data_name", type=str, default="simple-n1000-t10", help="simple-n1000-t10;complex-n1000-t10-p5-h5")
    parser_gen_data.set_defaults(func=gen_data)

    # make_table command
    parser_make_table = subparsers.add_parser('make_table', help='make summary tables')
    parser_make_table.add_argument("--table_name", type=str, default='simple', help="simple;complex")
    parser_make_table.set_defaults(func=make_table)

    # estimate command
    parser_estimate = subparsers.add_parser('estimate', help='estimate')
    parser_estimate.add_argument("--name", default="default_estimate")
    parser_estimate.add_argument("--seed", type=int, default=1234)
    parser_estimate.add_argument("--tau", nargs='+', type=int, default=None)
    parser_estimate.add_argument("--max_tau", type=int, default=None)
    parser_estimate.add_argument("--max_epochs", type=int, default=None)
    parser_estimate.add_argument("--data_name", type=str, default="circs-lacy-mort-medium", help="circs-lacy-mort-medium")
    parser_estimate.add_argument("--configuration_name", type=str, default="dltmle", help="dltmle")
    parser_estimate.add_argument("--artifact_path", type=str, default="artifact/")
    parser_estimate.add_argument("--use_cpu", action="store_true")
    parser_estimate.add_argument("--overwrite_model", action="store_true")
    parser_estimate.set_defaults(func=estimate)

    # plot command
    parser_plot = subparsers.add_parser('plot', help='plot survival curve')
    parser_plot.add_argument("--tau", type=int, default=10)    
    parser_plot.add_argument("--data_name", type=str, default="circs-lacy-mort-medium", help="circs-lacy-mort-medium")
    parser_plot.add_argument("--configuration_name", type=str, default="dltmle", help="dltmle")
    parser_plot.set_defaults(func=plot)

    args = dict(vars(parser.parse_args()))
    args.pop("func")(args)

if __name__ == "__main__":
    main()
