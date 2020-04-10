import nox


def install_reqs(session):
    session.install("--progress-bar", "off", "--no-cache-dir", "-r", "requirements.txt")
    session.install(
        "--progress-bar", "off", "--no-cache-dir", "-r", "dev-requirements.txt"
    )


def run_lint(session):
    session.run("black", "slp")
    session.run("black", "examples")
    session.run("black", "tests")
    session.run("flake8", "slp")
    session.run("flake8", "examples")
    session.run("flake8", "tests")


def run_typecheck(session):
    session.run("python", "-m", "mypy", "--config-file", "mypy.ini", "-p", "slp")


def run_tests(session):
    session.run("pytest", "-s", "--cov", "slp", "tests")


@nox.session
def lint(session):
    session.install("flake8")
    session.install("black")
    run_lint(session)


@nox.session
def typecheck(session):
    install_reqs(session)
    run_typecheck(session)


@nox.session
def tests(session):
    install_reqs(session)
    session.install(".")
    run_tests(session)


@nox.session(python=False)
def lintci(session):
    run_lint(session)


@nox.session(python=False)
def typecheckci(session):
    run_typecheck(session)


@nox.session(python=False)
def testsci(session):
    run_tests(session)
