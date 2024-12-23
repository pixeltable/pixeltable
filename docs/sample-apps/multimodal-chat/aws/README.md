# Deploy Multimodal API

This guide will help you deploy the Multimodal API to AWS using CDK. The stack provisions an ECS Fargate service with auto-scaling, load balancing, and secure secrets management.

## Limitations
This deployment has the following limitations:

- No Session Management: The API doesn't maintain user sessions. Each request is treated independently.
- No Ephemeral Storage Management: The container's storage isn't cleaned up automatically. Long-running containers will eventually run out of disk space if temp files accumulate.

Consider implementing:

- Session handling if user state is needed
- A cleanup routine for temporary filess
- Using EFS if persistent storage is required

## Prerequisites

Ensure you have the following installed and configured:

- AWS CLI with configured credentials
- AWS CDK v2
- Docker Desktop 
- Node.js 18+
- Python 3.8+
- An OpenAI API key

## Project Structure
```
.
├── aws/          # CDK infrastructure code
├── backend/      # API service code + Dockerfile
└── .env          # Environment variables
```

## Environment Setup

1. Create a `.env` file in the root directory:
```
OPENAI_API_KEY=your_key_here
```

2. Install dependencies:
```bash
# Install CDK dependencies
cd aws
npm install

# Install Python dependencies
pip install -r requirements.txt
```

## Deploy

From the `aws` directory:

```bash
cdk bootstrap && cdk deploy
```

This deploys:
- VPC with public/private subnets
- ECS Fargate cluster
- Application Load Balancer
- Auto-scaling policies
- Secrets in AWS Secrets Manager

The API endpoint URL will be output after deployment.

## Architecture

- **Compute**: ECS Fargate (ARM64) with 8 vCPU/16GB RAM
- **Scaling**: Auto-scales based on CPU, memory and request count
- **Monitoring**: Container insights and CloudWatch logs
- **Security**: Secrets stored in AWS Secrets Manager

## Cleanup

Remove all resources:

```bash
cdk destroy
```

## Costs
This deployment includes billable AWS services. Key cost factors:
- Fargate compute
- NAT Gateway 
- Application Load Balancer
- CloudWatch Logs