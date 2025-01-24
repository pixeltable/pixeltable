import os

from aws_cdk import (
    Stack,
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_ecs_patterns as ecs_patterns,
    aws_ecr_assets as ecr_assets,
    aws_logs as logs,
    aws_secretsmanager as secretsmanager,
    Duration,
    CfnOutput,
    SecretValue,
    App,
)

from constructs import Construct

from dotenv import load_dotenv

load_dotenv()


class MultimodalEcsStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Create secret manager
        secret = secretsmanager.Secret(
            self,
            'MultimodalSecret',
            secret_name='multimodal/api-secrets',
            secret_object_value={'OPENAI_API_KEY': SecretValue.unsafe_plain_text(os.getenv('OPENAI_API_KEY'))},
        )

        # Create VPC
        vpc = ec2.Vpc(
            self,
            'MultimodalVPC',
            max_azs=2,
            nat_gateways=1,
            subnet_configuration=[
                ec2.SubnetConfiguration(name='Public', subnet_type=ec2.SubnetType.PUBLIC, cidr_mask=24),
                ec2.SubnetConfiguration(name='Private', subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS, cidr_mask=24),
            ],
        )

        # Create ECS Cluster
        cluster = ecs.Cluster(self, 'MultimodalCluster', vpc=vpc, container_insights=True)

        # Create CloudWatch log group
        log_group = logs.LogGroup(self, 'MultimodalLogGroup', retention=logs.RetentionDays.ONE_WEEK)

        # Build Docker image from backend directory
        image = ecr_assets.DockerImageAsset(
            self, 'MultimodalImage', directory='../backend', platform=ecr_assets.Platform.LINUX_ARM64
        )

        # Create Fargate Service with ALB
        fargate_service = ecs_patterns.ApplicationLoadBalancedFargateService(
            self,
            'MultimodalService',
            cluster=cluster,
            cpu=8192,  # 8 vCPU
            memory_limit_mib=16384,  # 16 GB RAM (2GB per vCPU is recommended)
            desired_count=1,
            platform_version=ecs.FargatePlatformVersion.VERSION1_4,
            runtime_platform=ecs.RuntimePlatform(
                cpu_architecture=ecs.CpuArchitecture.ARM64, operating_system_family=ecs.OperatingSystemFamily.LINUX
            ),
            task_image_options=ecs_patterns.ApplicationLoadBalancedTaskImageOptions(
                image=ecs.ContainerImage.from_docker_image_asset(image),
                container_port=8000,  # Multimodal default port
                container_name='Multimodal-container',
                enable_logging=True,
                secrets={'OPENAI_API_KEY': ecs.Secret.from_secrets_manager(secret, 'OPENAI_API_KEY')},
                log_driver=ecs.LogDrivers.aws_logs(stream_prefix='Multimodal', log_group=log_group),
            ),
            public_load_balancer=True,
            listener_port=80,
        )

        # Add Auto Scaling
        scaling = fargate_service.service.auto_scale_task_count(
            min_capacity=1,
            max_capacity=10,  # Maximum number of tasks
        )

        # CPU utilization scaling
        scaling.scale_on_cpu_utilization(
            'CpuScaling',
            target_utilization_percent=70,
            scale_in_cooldown=Duration.seconds(60),
            scale_out_cooldown=Duration.seconds(60),
        )

        # Memory utilization scaling
        scaling.scale_on_memory_utilization(
            'MemoryScaling',
            target_utilization_percent=70,
            scale_in_cooldown=Duration.seconds(60),
            scale_out_cooldown=Duration.seconds(60),
        )

        # Request count per target scaling
        scaling.scale_on_request_count(
            'RequestCountScaling',
            requests_per_target=1000,
            target_group=fargate_service.target_group,
            scale_in_cooldown=Duration.seconds(60),
            scale_out_cooldown=Duration.seconds(60),
        )

        # Configure health check for ALB target group
        fargate_service.target_group.configure_health_check(
            path='/',
            healthy_http_codes='200,404',  # Including 404 as it might be a valid response
            interval=Duration.seconds(30),
            timeout=Duration.seconds(10),
            healthy_threshold_count=2,
            unhealthy_threshold_count=3,
        )

        # Add security group rule to allow inbound traffic
        fargate_service.service.connections.allow_from_any_ipv4(
            ec2.Port.tcp(8000), description='Allow inbound HTTP traffic'
        )

        # Output the ALB DNS name
        CfnOutput(
            self,
            'LoadBalancerDNS',
            value=fargate_service.load_balancer.load_balancer_dns_name,
            description='Load balancer DNS name',
        )


app = App()
MultimodalEcsStack(app, 'MultimodalApiStack')
app.synth()
