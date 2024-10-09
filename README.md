```sh
clearml-task --project clearml_test --name remote_test --repo https://github.com/Soundscape/clearml-test.git --script main.py --queue default --docker mcr.microsoft.com/devcontainers/python:1-3.12-bullseye
```