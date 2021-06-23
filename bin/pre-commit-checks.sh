#!/bin/bash
set -e

DOCKER_URL="metasearch-data-local:latest"
DOCKER_OPS=$1

get_script_dir () {
  SOURCE="${BASH_SOURCE[0]}"
  # While $SOURCE is a symlink, resolve it
  while [ -h "$SOURCE" ]; do
    DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
    SOURCE="$( readlink "$SOURCE" )"
    # If $SOURCE was a relative symlink (so no "/" as prefix, need to resolve it relative to the symlink base directory
    [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
  done
  DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
  echo "$DIR"
}

SCRIPT_DIR=$(get_script_dir)
PROJECT_DIR="${SCRIPT_DIR}"/..
DOCKER_DIR="${SCRIPT_DIR}"/docker

run_checks() {
	pushd ${PROJECT_DIR} > /dev/null
	echo -e "\nRunning flake8 checks ..."
	poetry run flake8 upliftml tests
	echo -e "\nRunning black checks ..."
	poetry run black --check upliftml tests
	echo -e "\nRunning isort checks ..."
	poetry run isort {upliftml,tests}/*.py --check-only
	echo -e "\nRunning mypy checks ..."
	poetry run mypy upliftml tests
	echo -e "\nRunning pytest ..."
	poetry run pytest
	popd > /dev/null
}

build_local_docker_image() {
  pushd ${PROJECT_DIR} && \
  docker build -t ${DOCKER_URL} -f ${DOCKER_DIR}/Dockerfile . && \
  popd
}

if [ "${DOCKER_OPS}" = "--docker" ]; then
  echo "Running checks in docker..."
  build_local_docker_image && \
  pushd ${PROJECT_DIR} > /dev/null && \
  docker run -v $(pwd):/metasearch-data -it ${DOCKER_URL} /bin/bash -c "cd /metasearch-data && bin/pre-commit-checks.sh" && \
  popd > /dev/null
else
  run_checks
fi

