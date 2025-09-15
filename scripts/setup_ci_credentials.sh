#!/bin/bash

# Replacee existing ~/.aws config and credentials files with values needed to access an R2 storage destination.
# Values are copied from the three environment variables:
# R2_ACCESS_KEY_ID
# R2_SECRET_ACCESS_KEY
# so that the r2_profile can be used when setting up boto3, via the python code:
#    session = boto3.Session(profile_name='r2_profile')

# Create AWS directory if needed
mkdir -p ~/.aws

# Remove any existing config or credentials files
rm -rf ~/.aws/config
rm -rf ~/.aws/credentials

# Add default and r2_profile to AWS config
cat > ~/.aws/config << EOF
[default]
EOF

# Add the r2_profile credentials to AWS credentials
cat > ~/.aws/credentials << EOF
[r2_profile]
aws_access_key_id = ${R2_ACCESS_KEY_ID}
aws_secret_access_key = ${R2_SECRET_ACCESS_KEY}
EOF

# Set permissions
chmod 600 ~/.aws/credentials
chmod 600 ~/.aws/config

# Setup GCS if credentials are provided
if [ ! -z "${GCS_SERVICE_ACCOUNT_KEY}" ]; then
    echo "${GCS_SERVICE_ACCOUNT_KEY}" > /tmp/gcs-key.json
    chmod 600 /tmp/gcs-key.json
fi
