#!/bin/bash -l

# Script to run a shell command several times with a delay between retries.
# This is useful for running flaky commands in CI, such as those that depend on
# network connectivity.
# Usage: retry.sh <num-retries> <sleep-duration> <command>

set +e

retries="$1"
sleep_interval="$2"
shift 2

if [[ -z "$@" ]]; then
    echo "Usage: retry.sh <num-retries> <sleep-duration> <command>"
    echo "Example: retry.sh 3 10 ollama pull 'qwen2.5:0.5b'"
    exit 1
fi

echo "Running command with $retries retries: $@"

while (( retries-- > 0)); do
    $@
    RESULT="$?"
    if [[ "$RESULT" == 0 ]]; then
        break
    fi
    echo "Failed with exit code $RESULT; $retries tries remaining."
    if (( retries > 0 )); then
        sleep $sleep_interval
    fi
done

if [[ "$RESULT" != 0 ]]; then
    echo "Command failed: $@"
    exit 1
fi

echo "Command succeeded: $@"
