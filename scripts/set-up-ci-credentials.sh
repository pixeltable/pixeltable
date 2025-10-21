#!/bin/bash -e

# Create AWS directory (should be empty at time this script is run)
mkdir ~/.aws

# Add default to AWS config
cat >> ~/.aws/config << EOF
[default]
region = ${AWS_DEFAULT_REGION}
EOF

# Add profiles to AWS credentials
if [ -n "${AWS_ACCESS_KEY_ID}" ]; then
cat >> ~/.aws/credentials << EOF
[default]
aws_access_key_id = ${AWS_ACCESS_KEY_ID}
aws_secret_access_key = ${AWS_SECRET_ACCESS_KEY}

EOF
fi

if [ -n "${R2_ACCESS_KEY_ID}" ]; then
cat >> ~/.aws/credentials << EOF
[r2_profile]
aws_access_key_id = ${R2_ACCESS_KEY_ID}
aws_secret_access_key = ${R2_SECRET_ACCESS_KEY}

EOF
fi

if [ -n "${B2_ACCESS_KEY_ID}" ]; then
cat >> ~/.aws/credentials << EOF
[b2_profile]
aws_access_key_id = ${B2_ACCESS_KEY_ID}
aws_secret_access_key = ${B2_SECRET_ACCESS_KEY}

EOF
fi

# Set permissions
chmod a+r ~/.aws/config
chmod a+r ~/.aws/credentials || true

# Setup GCS if credentials are provided
if [ -n "${GCS_SERVICE_ACCOUNT_KEY}" ]; then
    echo "${GCS_SERVICE_ACCOUNT_KEY}" > /tmp/gcs-key.json
    chmod a+r /tmp/gcs-key.json
fi
