import nox

def install_reqs(session):
    session.install('--progress-bar', 'off', '--no-cache-dir', '-r', 'requirements.txt')
    session.install('--progress-bar', 'off', '--no-cache-dir', '-r', 'dev-requirements.txt')
 

@nox.session
def lint(session):
    session.install('flake8')
    session.run('flake8', 'slp')
    session.run('flake8', 'examples')
    session.run('flake8', 'tests')


@nox.session
def typecheck(session):
    install_reqs(session)
    session.run('python', '-m', 'mypy', '--config-file', 'mypy.ini', '-p', 'slp')

@nox.session
def tests(session):
    install_reqs(session)
    session.install('.')
    session.run('pytest', '-s', '--cov', 'slp', 'tests')
