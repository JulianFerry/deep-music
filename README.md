# Data Science Project Template

## Objective

The motive of this project template is to start data-science projects with deployment in mind.

Everything is modular and therefore modifiable. By default:

- `Poetry` is used for dependency management
- `Docker` is used to spin up local development servers (e.g. databases)
- `Gitlab CI` is used to run continuous integration tests and to build artifacts
- `Docker` images are the target artifact for production
- The core python stack is `sqlalchemy`, `pandas`, `scikit-learn`, `joblib` and `flask`

Once the project grows, the template will have to be adapted - e.g. by splitting each stage of the ML pipeline (data ingest, preprocessing, hyperparameter tuning, model training, model serving) into multiple packages, each with their own set of dependencies and Dockerfiles.

## Requirements

- python ^3.7
- poetry ^1.0.5 - install with `curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python`
- docker ^18.06 (optional) to deploy build images

# Development setup

1. Clone this repository, modify the project name in `pyproject.toml` and run `poetry install --no-root --extras eda` to install all dependencies. This installs:

   - sqlalchemy, pandas, scikit-learn, flask and joblib as core dependencies
   - jupyter and seaborn as extra EDA dependencies
   - flake8, pytest and pytest-cov as dev dependencies

2. New packages can be installed with `poetry add package_name`. This will install the new packages to the `.venv` virtual environment and automatically update the `pyproject.toml` and `poetry.lock` files to capture the new project dependencies.

## Development to production workflow

This template is designed with the following workflow in mind:

### I. Local development

1. **Data:** Decide on a data storage solution for development:

   - **Working with .csv files:** Modify the functions in `database.py` to load data from disk.
   - **Local database setup:** Run `docker/db/docker-run.zsh` to launch a msyql database on port 3306.
   - **Remote database connection:** Set your environment variables as seen in the `database.py` `get_config()` function.

2. **Dev:** Develop in jupyter in `notebooks/` (use database `dev` schema)

3. **Dev:** Move code to python modules in `src/` (use database `dev` schema).
   - **Note:** Each ML pipeline stage (data_ingest, preprocessing and model training) should should eventually become standalone packages. Each will have a build artifact which can be run on a powerful cloud server.

4. **Test:** Run tests locally with `poetry run pytest` (you should monkeypatch db calls)

5. **Test/CI:** Run tests on a local gitlab-ci runner with `source docker/ci_tests/docker-run.zsh` (this also allows you to validate your .gitlab-ci.yml file before performing a git commit or push)

### II. Continuous Integration / Builds / Deployment

After checking that tests pass on a gitlab-runner:

6. **CI**: Once you `git push`, Gitlab CI will read `.gitlab-ci.yml` and should perform the following for each package:
   1. **Test:** Run tests against the code
   2. **Build:** Bundle the code into an artifact (python package or docker image)
   3. **Release:** Push the artifact to a registry (pypi server or docker registry)

7. **Staging:** Run the artifacts in a replica of the production environment (use database `staging` schema)
   1. **Test:** Run automated integration tests.
   2. **Test:** A third party such as the test team runs manual tests

8. **Prod:** Semi-automated deployment of the artifact on a production server (use database `prod` schema)
