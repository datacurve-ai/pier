from typer.main import get_command

from pier.cli.main import app


def test_run_uses_optimize_concurrency_flag():
    root_command = get_command(app)
    run_command = root_command.commands["run"]
    option_names = {
        option_name
        for parameter in run_command.params
        for option_name in parameter.opts
    }

    assert "--optimize_concurrency" in option_names
    assert "--dynamic-concurrency" not in option_names
