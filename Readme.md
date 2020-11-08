# Marketing bank analysis
Dataset available [here](https://www.kaggle.com/prakharrathi25/banking-dataset-marketing-targets)

# Run webapp using docker
## Create the docker image
```sh
docker build -t bank_marketing_targets .
```

## Run Container from the image
```sh
docker run -p 8050:8050 bank_marketing_targets
```