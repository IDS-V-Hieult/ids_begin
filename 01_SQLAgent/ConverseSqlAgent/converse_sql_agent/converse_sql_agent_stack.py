from aws_cdk import (
    Stack,
    aws_dynamodb as dynamodb,
    aws_ec2 as ec2,
    aws_rds as rds,
    aws_secretsmanager as secretsmanager,
    aws_lambda as lambda_,
    aws_iam as iam,
    RemovalPolicy,
    Duration,
    Size,
    Tags
)
import aws_cdk as cdk
from constructs import Construct

from cdk_nag import ( AwsSolutionsChecks, NagSuppressions )

class ConverseSqlAgentStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Create new VPC without NAT Gateways
        # Note: This is a cost-optimized architecture without NAT Gateways
        # Lambda functions in private subnets will use VPC endpoints for AWS services
        vpc = ec2.Vpc(
            self, "SQLAgentVPC",
            vpc_name="SQLAgent-vpc",
            ip_addresses=ec2.IpAddresses.cidr("10.2.0.0/16"),
            max_azs=2,
            nat_gateways=0,  # No NAT Gateways
            subnet_configuration=[]  # We'll create subnets manually
        )
        cdk.Tags.of(vpc).add("Name", "SQLAgent-vpc")

        # Get availability zones
        azs = vpc.availability_zones

        # Create public subnets
        public_subnet_01 = ec2.PublicSubnet(
            self, "SQLAgentPublic01Subnet",
            vpc_id=vpc.vpc_id,
            availability_zone=azs[0],
            cidr_block="10.2.1.0/24",
            map_public_ip_on_launch=True
        )
        cdk.Tags.of(public_subnet_01).add("Name", "SQLAgent-public01-subnet")

        public_subnet_02 = ec2.PublicSubnet(
            self, "SQLAgentPublic02Subnet",
            vpc_id=vpc.vpc_id,
            availability_zone=azs[1],
            cidr_block="10.2.2.0/24",
            map_public_ip_on_launch=True
        )
        cdk.Tags.of(public_subnet_02).add("Name", "SQLAgent-public02-subnet")

        # Create private subnets
        private_subnet_01 = ec2.PrivateSubnet(
            self, "SQLAgentPrivate01Subnet",
            vpc_id=vpc.vpc_id,
            availability_zone=azs[0],
            cidr_block="10.2.3.0/24"
        )
        cdk.Tags.of(private_subnet_01).add("Name", "SQLAgent-private01-subnet")

        private_subnet_02 = ec2.PrivateSubnet(
            self, "SQLAgentPrivate02Subnet",
            vpc_id=vpc.vpc_id,
            availability_zone=azs[1],
            cidr_block="10.2.4.0/24"
        )
        cdk.Tags.of(private_subnet_02).add("Name", "SQLAgent-private02-subnet")

        # Create Internet Gateway
        igw = ec2.CfnInternetGateway(self, "SQLAgentIGW")
        cdk.Tags.of(igw).add("Name", "SQLAgent-igw")
        
        # Attach IGW to VPC
        ec2.CfnVPCGatewayAttachment(
            self, "SQLAgentIGWAttachment",
            vpc_id=vpc.vpc_id,
            internet_gateway_id=igw.ref
        )

        # Create Route Tables
        # Public Route Table
        public_route_table = ec2.CfnRouteTable(
            self, "SQLAgentPublicRouteTable",
            vpc_id=vpc.vpc_id
        )
        cdk.Tags.of(public_route_table).add("Name", "SQLAgent-rtb-public")

        # Private Route Tables
        private_route_table_01 = ec2.CfnRouteTable(
            self, "SQLAgentPrivateRouteTable01",
            vpc_id=vpc.vpc_id
        )
        cdk.Tags.of(private_route_table_01).add("Name", "SQLAgent-rtb-private01")

        private_route_table_02 = ec2.CfnRouteTable(
            self, "SQLAgentPrivateRouteTable02",
            vpc_id=vpc.vpc_id
        )
        cdk.Tags.of(private_route_table_02).add("Name", "SQLAgent-rtb-private02")

        # Add route to Internet Gateway for public route table
        ec2.CfnRoute(
            self, "PublicRoute",
            route_table_id=public_route_table.ref,
            destination_cidr_block="0.0.0.0/0",
            gateway_id=igw.ref
        )

        # Associate public subnets with public route table
        ec2.CfnSubnetRouteTableAssociation(
            self, "PublicSubnet01Association",
            subnet_id=public_subnet_01.subnet_id,
            route_table_id=public_route_table.ref
        )

        ec2.CfnSubnetRouteTableAssociation(
            self, "PublicSubnet02Association",
            subnet_id=public_subnet_02.subnet_id,
            route_table_id=public_route_table.ref
        )

        # Associate private subnets with their respective route tables
        ec2.CfnSubnetRouteTableAssociation(
            self, "PrivateSubnet01Association",
            subnet_id=private_subnet_01.subnet_id,
            route_table_id=private_route_table_01.ref
        )

        ec2.CfnSubnetRouteTableAssociation(
            self, "PrivateSubnet02Association",
            subnet_id=private_subnet_02.subnet_id,
            route_table_id=private_route_table_02.ref
        )

        # Create Network ACL
        # Note: This is a permissive NACL configuration for POC purposes
        # In production, consider implementing more restrictive rules
        network_acl = ec2.CfnNetworkAcl(
            self, "SQLAgentNetworkAcl",
            vpc_id=vpc.vpc_id
        )
        cdk.Tags.of(network_acl).add("Name", "SQLAgent-nacl")

        # Create Inbound Rule - Allow All Traffic
        # Rule 100: Allow all protocols, all ports from anywhere
        ec2.CfnNetworkAclEntry(
            self, "SQLAgentNaclInboundRule",
            network_acl_id=network_acl.ref,
            rule_number=100,
            protocol=-1,  # -1 means all protocols
            rule_action="allow",
            cidr_block="0.0.0.0/0",
            egress=False  # False means inbound
        )

        # Create Outbound Rule - Allow All Traffic
        # Rule 100: Allow all protocols, all ports to anywhere
        ec2.CfnNetworkAclEntry(
            self, "SQLAgentNaclOutboundRule",
            network_acl_id=network_acl.ref,
            rule_number=100,
            protocol=-1,  # -1 means all protocols
            rule_action="allow",
            cidr_block="0.0.0.0/0",
            egress=True  # True means outbound
        )

        # Associate Network ACL with all subnets
        # This replaces the default NACL for all subnets
        all_subnets = [public_subnet_01, public_subnet_02, private_subnet_01, private_subnet_02]
        for i, subnet in enumerate(all_subnets):
            ec2.CfnSubnetNetworkAclAssociation(
                self, f"SQLAgentNaclAssociation{i+1}",
                subnet_id=subnet.subnet_id,
                network_acl_id=network_acl.ref
            )

        # Store subnets for later use
        private_subnets = [private_subnet_01, private_subnet_02]

        # Create DynamoDB table
        dynamodb_table = dynamodb.Table(
            self, "TEXT2SQLTable",
            table_name="advtext2sql_memory_tb",
            partition_key=dynamodb.Attribute(name="id", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.DESTROY
        )

        # Create RDS PostgreSQL instance
        # First create the secret with PostgreSQL specific settings
        db_secret = secretsmanager.Secret(
            self, "DBSecret",
            generate_secret_string=secretsmanager.SecretStringGenerator(
                secret_string_template='{"username": "SQLAgentDBAdmin"}',
                generate_string_key="password",
                password_length=32,
                exclude_punctuation=False,
                include_space=False
            )
        )

        # Create DB subnet group
        db_subnet_group = rds.SubnetGroup(
            self, "SQLAgentDBSubnetGroup",
            vpc=vpc,
            description="Subnet group for SQLAgent RDS database",
            vpc_subnets=ec2.SubnetSelection(subnets=private_subnets),
            subnet_group_name="sqlagent-db-subnet-group"
        )
        
        # Create security group for RDS
        db_security_group = ec2.SecurityGroup(
            self, "SQLAgentDBSecurityGroup",
            vpc=vpc,
            security_group_name="sqlagent-db-sg",
            allow_all_outbound=False,  # Database doesn't need outbound internet
            description="Security group for SQLAgent RDS PostgreSQL database"
        )
        
        # Create security group for Lambda
        lambda_security_group = ec2.SecurityGroup(
            self, "SQLAgentLambdaSecurityGroup",
            vpc=vpc,
            allow_all_outbound=False,  # No outbound internet access needed
            description="Security group for SQLAgent Lambda function"
        )
        
        # Allow Lambda to connect to VPC endpoints and RDS
        lambda_security_group.add_egress_rule(
            peer=ec2.Peer.ipv4(vpc.vpc_cidr_block),
            connection=ec2.Port.all_traffic(),
            description="Allow communication within VPC"
        )
        
        # Allow Lambda to connect to RDS PostgreSQL
        db_security_group.add_ingress_rule(
            peer=lambda_security_group,
            connection=ec2.Port.tcp(5432),  # PostgreSQL port
            description="Allow Lambda to connect to PostgreSQL"
        )
        
        # Create RDS PostgreSQL instance
        db_instance = rds.DatabaseInstance(
            self, "SQLAgentDB",
            instance_identifier="SQLAgent-db",
            engine=rds.DatabaseInstanceEngine.postgres(version=rds.PostgresEngineVersion.VER_17_4),
            instance_type=ec2.InstanceType.of(ec2.InstanceClass.T3, ec2.InstanceSize.MICRO),
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(subnets=private_subnets),
            subnet_group=db_subnet_group,
            credentials=rds.Credentials.from_secret(db_secret),
            database_name="sqlagentdb",
            multi_az=False,
            allocated_storage=20,
            storage_type=rds.StorageType.GP3,
            max_allocated_storage=100,
            security_groups=[db_security_group],
            publicly_accessible=False,
            backup_retention=Duration.days(7),
            delete_automated_backups=True,
            deletion_protection=False,
            removal_policy=RemovalPolicy.DESTROY,
            auto_minor_version_upgrade=True
        )

        # Create VPC Endpoints
        # Note: Since we don't have NAT Gateways, Lambda functions in private subnets
        # need VPC endpoints to access AWS services
        dynamodb_endpoint = vpc.add_gateway_endpoint(
            "DynamoDBEndpoint",
            service=ec2.GatewayVpcEndpointAwsService.DYNAMODB,
            subnets=[ec2.SubnetSelection(subnets=private_subnets)]
        )
        
        secrets_manager_endpoint = vpc.add_interface_endpoint(
            "SecretsManagerEndpoint",
            service=ec2.InterfaceVpcEndpointAwsService.SECRETS_MANAGER,
            subnets=ec2.SubnetSelection(subnets=private_subnets),
            security_groups=[lambda_security_group]
        )
        
        bedrock_endpoint = vpc.add_interface_endpoint(
            "BedrockEndpoint",
            service=ec2.InterfaceVpcEndpointAwsService.BEDROCK_RUNTIME,
            subnets=ec2.SubnetSelection(subnets=private_subnets),
            security_groups=[lambda_security_group]
        )

        # Create Lambda function
        lambda_role = iam.Role(
            self, "LambdaRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com")
        )

        # Add permissions for DynamoDB, Secrets Manager, and Bedrock
        
        # Grant DynamoDB permissions
        dynamodb_table.grant_read_write_data(lambda_role)

        # Grant Secrets Manager permissions
        db_secret.grant_read(lambda_role)
        
        lambda_role.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaVPCAccessExecutionRole"))
        lambda_role.add_to_policy(iam.PolicyStatement(
            actions=["bedrock:InvokeModel"],
            resources=[f"arn:aws:bedrock:{self.region}::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0"]
        ))

        # Create Lambda layers
        layer1 = lambda_.LayerVersion(
            self, "psycopg2_final",
            layer_version_name="layer_content",
            code=lambda_.Code.from_asset("./src/layers/layer_content.zip"),
            compatible_runtimes=[lambda_.Runtime.PYTHON_3_11]
        )        

        # Create Lambda function
        lambda_function = lambda_.Function(
            self, "SQLAgentFunction",
            function_name="sqlagent",
            runtime=lambda_.Runtime.PYTHON_3_11,
            handler="lambda_function.lambda_handler",
            code=lambda_.Code.from_asset("./src/ConverseSqlAgent"),
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(subnets=private_subnets),
            security_groups=[lambda_security_group],
            layers=[layer1],
            role=lambda_role,
            memory_size=1024,
            ephemeral_storage_size=Size.gibibytes(2),
            timeout=Duration.minutes(15),
            environment={
                "DynamoDbMemoryTable": dynamodb_table.table_name,
                "BedrockModelId": "anthropic.claude-3-sonnet-20240229-v1:0"
            }
        )

        # Grant permissions
        dynamodb_table.grant_read_write_data(lambda_function)
        db_secret.grant_read(lambda_function)