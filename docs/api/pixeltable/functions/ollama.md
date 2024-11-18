Pixeltable integrates with the popular [Ollama](https://ollama.com/) model server. To use these endpoints, you need to
either have an Ollama server running locally, or explicitly specify an Ollama host in your Pixeltable configration.
To specify an explicit host, either set the `OLLAMA_HOST` environment variable, or add an entry for `host` in the
`ollama` section of your `$PIXELTABLE_HOME/config.toml` configuration file.

## ::: pixeltable.functions.ollama
