#!/bin/bash
# Reference launch commands. Review before running; this creates billable resources.
# Prereqs: AWS CLI v2 logged in to account with Budgets alert set. Region us-east-1.
set -euo pipefail
REGION=us-east-1
TYPE="${INSTANCE_TYPE:-r7i.large}"          # 2 vCPU / 16 GB. Alternative: m7i.xlarge (4 vCPU / 16 GB)
ROLE=apply-or-skip-ec2
aws iam create-role --role-name $ROLE --assume-role-policy-document file://deploy/iam-trust-ec2.json
aws iam put-role-policy --role-name $ROLE --policy-name bedrock-nova-only --policy-document file://deploy/iam-policy-bedrock-nova.json
aws iam attach-role-policy --role-name $ROLE --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
aws iam create-instance-profile --instance-profile-name $ROLE
aws iam add-role-to-instance-profile --instance-profile-name $ROLE --role-name $ROLE
sleep 10
AMI=$(aws ssm get-parameter --region $REGION --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 --query Parameter.Value --output text)
# No inbound rules: access is via SSM Session Manager only.
SG=$(aws ec2 create-security-group --region $REGION --group-name apply-or-skip-no-inbound --description "no inbound" --query GroupId --output text)
aws ec2 run-instances --region $REGION --image-id "$AMI" --instance-type "$TYPE" \
  --iam-instance-profile Name=$ROLE --security-group-ids "$SG" \
  --instance-initiated-shutdown-behavior stop \
  --metadata-options HttpTokens=required \
  --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=40,VolumeType=gp3,DeleteOnTermination=true}' \
  --user-data file://deploy/user-data.sh \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=apply-or-skip},{Key=project,Value=strands-hackathon}]' \
  --query 'Instances[0].InstanceId' --output text
