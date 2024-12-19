# Deploy Multimodal API

This guide will help you deploy the Multimodal API to AWS.
## Prerequisites

Ensure you have the following installed and configured:

- AWS CLI
- AWS CDK
- Docker
- Node JS
- Python

## Deploy

Ensure you are in the `aws` directory.

```bash
cdk bootstrap && cdk deploy
```

This will deploy the stack to AWS.

## Cleanup

To clean up the stack, run the following command:

```bash
cdk destroy
```
