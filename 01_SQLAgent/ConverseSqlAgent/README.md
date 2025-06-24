### Converse SQL Agent: Building an intelligent text-to-SQL agent using Amazon Bedrock and Converse API. This is a Sample Code for a quick POC and the infrastructure can be adapted based on final implementation.

**Note**: This architecture is designed for POC/Development environments with cost optimization in mind. Network ACLs are configured with permissive rules. For production use, implement stricter security controls.

Authors: Pavan Kumar, Parag Srivastava and Abdullah Siddiqui 

Converse SQL Agent is a simple and powerful text-to-sql solution that can connect to
different databases and queries them all through natural language. It is built using
Amazon Bedrock, Converse API, and a custom agent implementation that enables it to
plan, execute, and learn as you use it.

### This section contains details on how to run the setup using CDK code written in Python.

### Prerequisites 

1. Ensure CDK is installed and [bootstrap](https://docs.aws.amazon.com/cdk/v2/guide/bootstrapping-env.html) the environment as required to connect it to your AWS account. 
```
$ npm install -g aws-cdk
```

2. Check for existence of Python 3.11. 

```
python 3.11 --version
```

3. If the response is "python: command not found". Install Python 3.11 and pip for Python 3.11
```
sudo dnf install python3.11 -y
sudo dnf install python3.11-pip -y
```

### To run example

Clone the code to your working environment. Follow the steps as mentioned. 

1. Change your working directory to the downloaded folder. 
```
$ cd ConverseSqlAgent
```

2. Run the following Script to download the dependencies required by the lambda layer and creating a zip file to be utilized as a layer for the lambda code. 
```
cd ./src/layers
python3.11 -m venv create_layer
source create_layer/bin/activate
pip install -r requirements.txt
mkdir python
cp -r create_layer/lib python/
zip -r layer_content.zip python
deactivate
cd ../../
```

3. Once back in the working directory of your code. Create a Python virtual environment
```
python3.11 -m venv .venv
```

4. Activate virtual environment

_On MacOS or Linux_
```
source .venv/bin/activate
```

_On Windows_
```
.venv\Scripts\activate.bat
```

5. Install the required dependencies.

```
pip install -r requirements.txt
```

6. Synthesize (`cdk synth`) or deploy (`cdk deploy`) the example

```
cdk deploy
```

### To dispose of the stack afterwards:

```
cdk destroy
```

7. Once the resources are built. Connect to the PostgreSQL RDS instance and run your database schema.

Ensure that you have the psql client installed and that the RDS PostgreSQL instance security group allows inbound traffic on port 5432.

8. First, retrieve the connection details. Retrieve the username, password and database hostname values from Secrets Manager.

```
# Install PostgreSQL client if not already installed
# On Amazon Linux / RHEL:
sudo yum install postgresql15 -y

# On Ubuntu/Debian:
sudo apt-get install postgresql-client -y

# On macOS:
brew install postgresql
```

9. Connect to database:

```
psql -h <database hostname> -U SQLAgentDBAdmin -d sqlagentdb -p 5432
```

10. After connecting, you can create your database schema and check the tables.
```
# List schemas
\dn

# List tables
\dt

# Create a sample HR schema (you can use any PostgreSQL compatible schema)
CREATE SCHEMA IF NOT EXISTS hr;
SET search_path TO hr;

# Create your tables here...
```

11. You can test the lambda using the simple prompt as given below. 

```
{
  "input_text": "Connect to the database sqlagentdb using the secrets manager key <Database key> and get me how many employees are there in each department in each region?"
}
```

### Network Architecture

The solution creates a custom VPC with the following configuration:

- **VPC**: SQLAgent-vpc (10.2.0.0/16)
- **Subnets**:
  - Public Subnet 1: SQLAgent-public01-subnet (10.2.1.0/24)
  - Public Subnet 2: SQLAgent-public02-subnet (10.2.2.0/24)
  - Private Subnet 1: SQLAgent-private01-subnet (10.2.3.0/24)
  - Private Subnet 2: SQLAgent-private02-subnet (10.2.4.0/24)
- **Route Tables**:
  - SQLAgent-rtb-public: Routes to Internet via SQLAgent-igw
  - SQLAgent-rtb-private01: Private route table for subnet 01 (no internet access)
  - SQLAgent-rtb-private02: Private route table for subnet 02 (no internet access)
- **Internet Gateway**: SQLAgent-igw
- **NAT Gateway**: None (Lambda and RDS in private subnets use VPC endpoints for AWS services)
- **Network ACL**: SQLAgent-nacl
  - Applied to all 4 subnets
  - Inbound Rule 100: Allow all traffic from 0.0.0.0/0
  - Outbound Rule 100: Allow all traffic to 0.0.0.0/0

### RDS PostgreSQL Configuration

- **Database Name**: SQLAgent-db
- **Engine**: PostgreSQL 17.4
- **Instance Class**: db.t3.micro (2 vCPU, 1GB RAM)
- **Storage**: 20 GiB GP3
- **Database Name**: sqlagentdb
- **Master Username**: SQLAgentDBAdmin
- **Backup Retention**: 7 days
- **Multi-AZ**: No
- **Subnet Group**: sqlagent-db-subnet-group
- **Security Group**: sqlagent-db-sg

### Manual Installation steps

**Important Note**: Since this architecture doesn't use NAT Gateways, Lambda functions in private subnets cannot access the internet directly. All AWS service calls must go through VPC endpoints.

**Security Note**: The Network ACL (SQLAgent-nacl) is configured with permissive rules allowing all traffic for POC purposes. In production environments, implement more restrictive NACL rules based on your security requirements.

1. You will need to create python layer with the following dependencies 
	- pymysql
	- sqlalchemy
	- psycopg2-binary

2. Deploy these Layers to AWS. 
3. You will need to deploy the Lambda using the code available under "/src/ConverseSqlAgent" and the layer built on the previous step. 
4. Configure Lambda to run in the custom VPC (SQLAgent-vpc) which has the connectivity to the RDS database, and the credentials stored on Secret Manager
   - Ensure Lambda is placed in the private subnets
   - Network ACL (SQLAgent-nacl) allows all traffic - adjust for production use
5. Configure the following VPC endpoints in the same VPC:
	- Bedrock Runtime (com.amazonaws..bedrock-runtime)
	- DynamoDB (com.amazonaws..dynamodb)
	- Secrets Manager (com.amazonaws..secretsmanager)

4. Ensure the Lambda execution role has permissions for:
	- Bedrock Converse and the Claude 3 Sonnet
	- DynamoDB table for use with the agent
	- Secrets Manager key to store the RDS credentials

5. Ensure Lambda has the following environment variables:
	- DynamoDbMemoryTable (advtext2sql_memory_tb)
	- BedrockModelId (anthropic.claude-3-sonnet-20240229-v1:0)

6. Ensure that Lambda/VPC endpoints/RDS security groups allow communication
7. Use the Lambda test function to test the setup.

### Security Best Practices for Production

If deploying to production, consider implementing:
1. **Network ACLs**: Replace the permissive rules with specific allow rules for required traffic only
2. **Security Groups**: Implement least privilege access between components
3. **VPC Flow Logs**: Enable for network traffic monitoring
4. **AWS PrivateLink**: For additional AWS service endpoints as needed
5. **Secrets Rotation**: Enable automatic rotation for RDS credentials
6. **Encryption**: Enable encryption at rest for RDS and DynamoDB