/**
* HR Schema for PostgreSQL
* This is Dummy Data and does not represent any organization
*/

-- Drop schema if exists and create new one
DROP SCHEMA IF EXISTS hr CASCADE;
CREATE SCHEMA hr;

-- Set search path to hr schema
SET search_path TO hr;

/* *************************************************************** 
***************************CREATING TABLES************************
**************************************************************** */

CREATE TABLE regions (
    region_id INTEGER NOT NULL,
    region_name VARCHAR(25),
    PRIMARY KEY (region_id)
);

CREATE TABLE countries (
    country_id CHAR(2) NOT NULL,
    country_name VARCHAR(40),
    region_id INTEGER NOT NULL,
    PRIMARY KEY (country_id),
    CONSTRAINT fk_countries_regions 
        FOREIGN KEY (region_id) 
        REFERENCES regions(region_id)
);

CREATE TABLE locations (
    location_id SERIAL NOT NULL,
    street_address VARCHAR(40),
    postal_code VARCHAR(12),
    city VARCHAR(30) NOT NULL,
    state_province VARCHAR(25),
    country_id CHAR(2) NOT NULL,
    PRIMARY KEY (location_id),
    CONSTRAINT fk_locations_countries 
        FOREIGN KEY (country_id) 
        REFERENCES countries(country_id)
);

CREATE TABLE departments (
    department_id INTEGER NOT NULL,
    department_name VARCHAR(30) NOT NULL,
    manager_id INTEGER,
    location_id INTEGER,
    PRIMARY KEY (department_id),
    CONSTRAINT fk_departments_locations 
        FOREIGN KEY (location_id) 
        REFERENCES locations(location_id)
);

CREATE TABLE jobs (
    job_id VARCHAR(10) NOT NULL,
    job_title VARCHAR(35) NOT NULL,
    min_salary DECIMAL(8, 0),
    max_salary DECIMAL(8, 0),
    PRIMARY KEY (job_id)
);

CREATE TABLE employees (
    employee_id INTEGER NOT NULL,
    first_name VARCHAR(20),
    last_name VARCHAR(25) NOT NULL,
    email VARCHAR(25) NOT NULL,
    phone_number VARCHAR(20),
    hire_date DATE NOT NULL,
    job_id VARCHAR(10) NOT NULL,
    salary DECIMAL(8, 2) NOT NULL,
    commission_pct DECIMAL(2, 2),
    manager_id INTEGER,
    department_id INTEGER,
    PRIMARY KEY (employee_id),
    CONSTRAINT fk_employees_jobs 
        FOREIGN KEY (job_id) 
        REFERENCES jobs(job_id),
    CONSTRAINT fk_employees_departments 
        FOREIGN KEY (department_id) 
        REFERENCES departments(department_id),
    CONSTRAINT fk_employees_managers 
        FOREIGN KEY (manager_id) 
        REFERENCES employees(employee_id)
);

CREATE TABLE job_history (
    employee_id INTEGER NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    job_id VARCHAR(10) NOT NULL,
    department_id INTEGER NOT NULL,
    CONSTRAINT pk_job_history 
        PRIMARY KEY (employee_id, start_date),
    CONSTRAINT fk_job_history_employees 
        FOREIGN KEY (employee_id) 
        REFERENCES employees(employee_id),
    CONSTRAINT fk_job_history_jobs 
        FOREIGN KEY (job_id) 
        REFERENCES jobs(job_id),
    CONSTRAINT fk_job_history_departments 
        FOREIGN KEY (department_id) 
        REFERENCES departments(department_id)
);

-- Create view
CREATE VIEW emp_details_view AS
SELECT e.employee_id,
    e.job_id,
    e.manager_id,
    e.department_id,
    d.location_id,
    l.country_id,
    e.first_name,
    e.last_name,
    e.salary,
    e.commission_pct,
    d.department_name,
    j.job_title,
    l.city,
    l.state_province,
    c.country_name,
    r.region_name
FROM employees e
JOIN departments d ON e.department_id = d.department_id
JOIN jobs j ON j.job_id = e.job_id
JOIN locations l ON d.location_id = l.location_id
JOIN countries c ON l.country_id = c.country_id
JOIN regions r ON c.region_id = r.region_id;

/* *************************************************************** 
***************************INSERTING DATA*************************
**************************************************************** */

-- Insert regions
INSERT INTO regions VALUES (1, 'Europe');
INSERT INTO regions VALUES (2, 'Americas');
INSERT INTO regions VALUES (3, 'Asia');
INSERT INTO regions VALUES (4, 'Middle East and Africa');

-- Insert countries
INSERT INTO countries VALUES ('IT', 'Italy', 1);
INSERT INTO countries VALUES ('JP', 'Japan', 3);
INSERT INTO countries VALUES ('US', 'United States of America', 2);
INSERT INTO countries VALUES ('CA', 'Canada', 2);
INSERT INTO countries VALUES ('CN', 'China', 3);
INSERT INTO countries VALUES ('IN', 'India', 3);
INSERT INTO countries VALUES ('AU', 'Australia', 3);
INSERT INTO countries VALUES ('ZW', 'Zimbabwe', 4);
INSERT INTO countries VALUES ('SG', 'Singapore', 3);
INSERT INTO countries VALUES ('UK', 'United Kingdom', 1);
INSERT INTO countries VALUES ('FR', 'France', 1);
INSERT INTO countries VALUES ('DE', 'Germany', 1);
INSERT INTO countries VALUES ('ZM', 'Zambia', 4);
INSERT INTO countries VALUES ('EG', 'Egypt', 4);
INSERT INTO countries VALUES ('BR', 'Brazil', 2);
INSERT INTO countries VALUES ('CH', 'Switzerland', 1);
INSERT INTO countries VALUES ('NL', 'Netherlands', 1);
INSERT INTO countries VALUES ('MX', 'Mexico', 2);
INSERT INTO countries VALUES ('KW', 'Kuwait', 4);
INSERT INTO countries VALUES ('IL', 'Israel', 4);
INSERT INTO countries VALUES ('DK', 'Denmark', 1);
INSERT INTO countries VALUES ('HK', 'HongKong', 3);
INSERT INTO countries VALUES ('NG', 'Nigeria', 4);
INSERT INTO countries VALUES ('AR', 'Argentina', 2);
INSERT INTO countries VALUES ('BE', 'Belgium', 1);

-- Insert locations
INSERT INTO locations VALUES (DEFAULT, '1297 Via Cola di Rie', '00989', 'Roma', NULL, 'IT');
INSERT INTO locations VALUES (DEFAULT, '93091 Calle della Testa', '10934', 'Venice', NULL, 'IT');
INSERT INTO locations VALUES (DEFAULT, '2017 Shinjuku-ku', '1689', 'Tokyo', 'Tokyo Prefecture', 'JP');
INSERT INTO locations VALUES (DEFAULT, '9450 Kamiya-cho', '6823', 'Hiroshima', NULL, 'JP');
INSERT INTO locations VALUES (DEFAULT, '2014 Jabberwocky Rd', '26192', 'Southlake', 'Texas', 'US');
INSERT INTO locations VALUES (DEFAULT, '2011 Interiors Blvd', '99236', 'South San Francisco', 'California', 'US');
INSERT INTO locations VALUES (DEFAULT, '2007 Zagora St', '50090', 'South Brunswick', 'New Jersey', 'US');
INSERT INTO locations VALUES (DEFAULT, '2004 Charade Rd', '98199', 'Seattle', 'Washington', 'US');
INSERT INTO locations VALUES (DEFAULT, '147 Spadina Ave', 'M5V 2L7', 'Toronto', 'Ontario', 'CA');
INSERT INTO locations VALUES (DEFAULT, '6092 Boxwood St', 'YSW 9T2', 'Whitehorse', 'Yukon', 'CA');
INSERT INTO locations VALUES (DEFAULT, '40-5-12 Laogianggen', '190518', 'Beijing', NULL, 'CN');
INSERT INTO locations VALUES (DEFAULT, '1298 Vileparle (E)', '490231', 'Bombay', 'Maharashtra', 'IN');
INSERT INTO locations VALUES (DEFAULT, '12-98 Victoria Street', '2901', 'Sydney', 'New South Wales', 'AU');
INSERT INTO locations VALUES (DEFAULT, '198 Clementi North', '540198', 'Singapore', NULL, 'SG');
INSERT INTO locations VALUES (DEFAULT, '8204 Arthur St', NULL, 'London', NULL, 'UK');
INSERT INTO locations VALUES (DEFAULT, 'Magdalen Centre, The Oxford Science Park', 'OX9 9ZB', 'Oxford', 'Oxford', 'UK');
INSERT INTO locations VALUES (DEFAULT, '9702 Chester Road', '09629850293', 'Stretford', 'Manchester', 'UK');
INSERT INTO locations VALUES (DEFAULT, 'Schwanthalerstr. 7031', '80925', 'Munich', 'Bavaria', 'DE');
INSERT INTO locations VALUES (DEFAULT, 'Rua Frei Caneca 1360', '01307-002', 'Sao Paulo', 'Sao Paulo', 'BR');
INSERT INTO locations VALUES (DEFAULT, '20 Rue des Corps-Saints', '1730', 'Geneva', 'Geneve', 'CH');
INSERT INTO locations VALUES (DEFAULT, 'Murtenstrasse 921', '3095', 'Bern', 'BE', 'CH');
INSERT INTO locations VALUES (DEFAULT, 'Pieter Breughelstraat 837', '3029SK', 'Utrecht', 'Utrecht', 'NL');
INSERT INTO locations VALUES (DEFAULT, 'Mariano Escobedo 9991', '11932', 'Mexico City', 'Distrito Federal,', 'MX');

-- Insert departments (Note: We need to insert departments without manager_id first)
INSERT INTO departments VALUES (10, 'Administration', NULL, 8);
INSERT INTO departments VALUES (20, 'Marketing', NULL, 9);
INSERT INTO departments VALUES (30, 'Purchasing', NULL, 8);
INSERT INTO departments VALUES (40, 'Human Resources', NULL, 15);
INSERT INTO departments VALUES (50, 'Shipping', NULL, 6);
INSERT INTO departments VALUES (60, 'IT', NULL, 5);
INSERT INTO departments VALUES (70, 'Public Relations', NULL, 18);
INSERT INTO departments VALUES (80, 'Sales', NULL, 16);
INSERT INTO departments VALUES (90, 'Executive', NULL, 8);
INSERT INTO departments VALUES (100, 'Finance', NULL, 8);
INSERT INTO departments VALUES (110, 'Accounting', NULL, 8);
INSERT INTO departments VALUES (120, 'Treasury', NULL, 8);
INSERT INTO departments VALUES (130, 'Corporate Tax', NULL, 8);
INSERT INTO departments VALUES (140, 'Control And Credit', NULL, 8);
INSERT INTO departments VALUES (150, 'Shareholder Services', NULL, 8);
INSERT INTO departments VALUES (160, 'Benefits', NULL, 8);
INSERT INTO departments VALUES (170, 'Manufacturing', NULL, 8);
INSERT INTO departments VALUES (180, 'Construction', NULL, 8);
INSERT INTO departments VALUES (190, 'Contracting', NULL, 8);
INSERT INTO departments VALUES (200, 'Operations', NULL, 8);
INSERT INTO departments VALUES (210, 'IT Support', NULL, 8);
INSERT INTO departments VALUES (220, 'NOC', NULL, 8);
INSERT INTO departments VALUES (230, 'IT Helpdesk', NULL, 8);
INSERT INTO departments VALUES (240, 'Government Sales', NULL, 8);
INSERT INTO departments VALUES (250, 'Retail Sales', NULL, 8);
INSERT INTO departments VALUES (260, 'Recruiting', NULL, 8);
INSERT INTO departments VALUES (270, 'Payroll', NULL, 8);

-- Insert jobs
INSERT INTO jobs VALUES ('AD_PRES', 'President', 20000, 40000);
INSERT INTO jobs VALUES ('AD_VP', 'Administration Vice President', 15000, 30000);
INSERT INTO jobs VALUES ('AD_ASST', 'Administration Assistant', 3000, 6000);
INSERT INTO jobs VALUES ('FI_MGR', 'Finance Manager', 8200, 16000);
INSERT INTO jobs VALUES ('FI_ACCOUNT', 'Accountant', 4200, 9000);
INSERT INTO jobs VALUES ('AC_MGR', 'Accounting Manager', 8200, 16000);
INSERT INTO jobs VALUES ('AC_ACCOUNT', 'Public Accountant', 4200, 9000);
INSERT INTO jobs VALUES ('SA_MAN', 'Sales Manager', 10000, 20000);
INSERT INTO jobs VALUES ('SA_REP', 'Sales Representative', 6000, 12000);
INSERT INTO jobs VALUES ('PU_MAN', 'Purchasing Manager', 8000, 15000);
INSERT INTO jobs VALUES ('PU_CLERK', 'Purchasing Clerk', 2500, 5500);
INSERT INTO jobs VALUES ('ST_MAN', 'Stock Manager', 5500, 8500);
INSERT INTO jobs VALUES ('ST_CLERK', 'Stock Clerk', 2000, 5000);
INSERT INTO jobs VALUES ('SH_CLERK', 'Shipping Clerk', 2500, 5500);
INSERT INTO jobs VALUES ('IT_PROG', 'Programmer', 4000, 10000);
INSERT INTO jobs VALUES ('MK_MAN', 'Marketing Manager', 9000, 15000);
INSERT INTO jobs VALUES ('MK_REP', 'Marketing Representative', 4000, 9000);
INSERT INTO jobs VALUES ('HR_REP', 'Human Resources Representative', 4000, 9000);
INSERT INTO jobs VALUES ('PR_REP', 'Public Relations Representative', 4500, 10500);

-- Insert employees (simplified subset)
INSERT INTO employees VALUES (100, 'Alejandro', 'Rosalez', 'alejandro_rosalez', '515.123.4567', '1987-06-17', 'AD_PRES', 24000, NULL, NULL, 90);
INSERT INTO employees VALUES (101, 'Akua', 'Mansa', 'akua_mansa', '515.123.4568', '1989-09-21', 'AD_VP', 17000, NULL, 100, 90);
INSERT INTO employees VALUES (102, 'Ana Carolina', 'Silva', 'anacarolina_silva', '515.123.4569', '1993-01-13', 'AD_VP', 17000, NULL, 100, 90);
INSERT INTO employees VALUES (103, 'Arnav', 'Desai', 'arnav_desai', '590.423.4567', '1990-01-03', 'IT_PROG', 9000, NULL, 102, 60);
INSERT INTO employees VALUES (108, 'Jane', 'Doe', 'jane_doe', '515.124.4569', '1994-08-17', 'FI_MGR', 12000, NULL, 101, 100);
INSERT INTO employees VALUES (114, 'Kwesi', 'Manu', 'kwesi_manu', '515.127.4561', '1994-12-07', 'PU_MAN', 11000, NULL, 100, 30);
INSERT INTO employees VALUES (120, 'Mary', 'Major', 'mary_major', '650.123.1234', '1996-07-18', 'ST_MAN', 8000, NULL, 100, 50);
INSERT INTO employees VALUES (121, 'Mateo', 'Jackson', 'mateo_jackson', '650.123.2234', '1997-04-10', 'ST_MAN', 8200, NULL, 100, 50);
INSERT INTO employees VALUES (145, 'John', 'Stiles1', 'john_stiles1', '011.44.1344.429268', '1996-10-01', 'SA_MAN', 14000, 0.4, 100, 80);
INSERT INTO employees VALUES (200, 'Xiulan2', 'Wang2', 'xiulan_wang2', '515.123.4444', '1987-09-17', 'AD_ASST', 4400, NULL, 101, 10);
INSERT INTO employees VALUES (201, 'Wei2', 'Zhang2', 'wei_zhang2', '515.123.5555', '1996-02-17', 'MK_MAN', 13000, NULL, 100, 20);
INSERT INTO employees VALUES (203, 'Akua3', 'Mansa3', 'akua_mansa3', '515.123.7777', '1994-06-07', 'HR_REP', 6500, NULL, 101, 40);
INSERT INTO employees VALUES (204, 'Ana Carolina3', 'Silva3', 'anacarolina_silva3', '515.123.8888', '1994-06-07', 'PR_REP', 10000, NULL, 101, 70);
INSERT INTO employees VALUES (205, 'Arnav3', 'Desai3', 'arnav_desai3', '515.123.8080', '1994-06-07', 'AC_MGR', 12000, NULL, 101, 110);

-- Update departments with managers after employees are inserted
UPDATE departments SET manager_id = 200 WHERE department_id = 10;
UPDATE departments SET manager_id = 201 WHERE department_id = 20;
UPDATE departments SET manager_id = 114 WHERE department_id = 30;
UPDATE departments SET manager_id = 203 WHERE department_id = 40;
UPDATE departments SET manager_id = 121 WHERE department_id = 50;
UPDATE departments SET manager_id = 103 WHERE department_id = 60;
UPDATE departments SET manager_id = 204 WHERE department_id = 70;
UPDATE departments SET manager_id = 145 WHERE department_id = 80;
UPDATE departments SET manager_id = 100 WHERE department_id = 90;
UPDATE departments SET manager_id = 108 WHERE department_id = 100;
UPDATE departments SET manager_id = 205 WHERE department_id = 110;

-- Insert job history
INSERT INTO job_history VALUES (102, '1993-01-13', '1998-07-24', 'IT_PROG', 60);
INSERT INTO job_history VALUES (101, '1989-09-21', '1993-10-27', 'AC_ACCOUNT', 110);
INSERT INTO job_history VALUES (101, '1993-10-28', '1997-03-15', 'AC_MGR', 110);
INSERT INTO job_history VALUES (201, '1996-02-27', '1999-12-19', 'MK_REP', 20);
INSERT INTO job_history VALUES (114, '1998-03-24', '1999-12-31', 'ST_CLERK', 50);
INSERT INTO job_history VALUES (200, '1987-09-17', '1993-06-17', 'AD_ASST', 90);
INSERT INTO job_history VALUES (200, '1994-07-01', '1998-12-31', 'AC_ACCOUNT', 90);

-- Grant permissions (optional, depending on your use case)
-- GRANT ALL PRIVILEGES ON SCHEMA hr TO your_user;
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA hr TO your_user;
-- GRANT USAGE ON ALL SEQUENCES IN SCHEMA hr TO your_user;

-- Verify data
SELECT 'Regions count:', COUNT(*) FROM regions
UNION ALL
SELECT 'Countries count:', COUNT(*) FROM countries
UNION ALL
SELECT 'Locations count:', COUNT(*) FROM locations
UNION ALL
SELECT 'Departments count:', COUNT(*) FROM departments
UNION ALL
SELECT 'Jobs count:', COUNT(*) FROM jobs
UNION ALL
SELECT 'Employees count:', COUNT(*) FROM employees
UNION ALL
SELECT 'Job history count:', COUNT(*) FROM job_history;