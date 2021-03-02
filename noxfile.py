import nox


def install_reqs(session):
    session.install("--progress-bar", "off", "--no-cache-dir", "-r", "requirements.txt")


def run_lint(session):
    session.run("black", "--check", "slp")
    session.run("black", "--check", "examples")
    session.run("black", "--check", "tests")


def run_typecheck(session):
    session.run("python", "-m", "mypy", "--config-file", "mypy.ini", "-p", "slp")


def run_tests(session):
    session.run("pytest", "-s", "--cov", "slp", "tests")


@nox.session
def lint(session):
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
