#!/bin/bash
# Belt-and-braces: CloudWatch alarm that STOPS the instance after 30 min of CPU < 5%.
# Usage: INSTANCE_ID=i-... ./deploy/cloudwatch-idle-alarm.sh   (run from your laptop/CloudShell)
set -euo pipefail
: "${INSTANCE_ID:?set INSTANCE_ID}"
REGION="${AWS_REGION:-us-east-1}"
aws cloudwatch put-metric-alarm --region "$REGION" \
  --alarm-name "apply-or-skip-idle-stop-$INSTANCE_ID" \
  --namespace AWS/EC2 --metric-name CPUUtilization --dimensions Name=InstanceId,Value="$INSTANCE_ID" \
  --statistic Average --period 300 --evaluation-periods 6 --threshold 5 --comparison-operator LessThanThreshold \
  --alarm-actions "arn:aws:automate:$REGION:ec2:stop"
