import nox

@nox.session
def lint(session):
    session.install('flake8')
    session.run('flake8', 'slp')
    session.run('flake8', 'examples')
    session.run('flake8', 'tests')


@nox.session
def typecheck(session):
    session.install('-r', 'requirements.txt')
    session.install('-r', 'dev-requirements.txt')
    session.run('python', '-m', 'mypy', '--config-file', 'mypy.ini', '-p', 'slp')

@nox.session(python=['3.6', '3.7'])
def tests(session):
    session.install('-r', 'requirements.txt')
    session.install('-r', 'dev-requirements.txt')
    session.install('.')
    session.run('pytest', '-s', '--cov', 'slp', 'tests')
