-- queries.sql
-- Business-question SQL queries against the cleaned HR attrition data.
-- Each query independently validates a finding from the Python/notebook analysis
-- (see notebooks/01_eda.ipynb and README.md for the full narrative).
--
-- To run: import data/hr_clean.csv into DB Browser for SQLite as a table
-- named "employees" (File > Import > Table from CSV file), then paste these
-- into the Execute SQL tab.

-- 1. Sanity check: overall headcount and attrition rate
SELECT
    COUNT(*) AS total_employees,
    SUM(AttritionFlag) AS total_leavers,
    ROUND(100.0 * SUM(AttritionFlag) / COUNT(*), 1) AS attrition_rate_pct
FROM employees;
-- Expected: 1470 employees, ~16.1% attrition rate

-- 2. Attrition rate by department
SELECT
    Department,
    COUNT(*) AS headcount,
    SUM(AttritionFlag) AS leavers,
    ROUND(100.0 * SUM(AttritionFlag) / COUNT(*), 1) AS attrition_rate_pct
FROM employees
GROUP BY Department
ORDER BY attrition_rate_pct DESC;
-- Finding: Sales (20.6%) and HR (19.0%) lose people at ~1.5x the rate of R&D (13.8%)

-- 3. Overtime's effect on attrition, broken down by department
SELECT
    Department,
    OverTime,
    COUNT(*) AS headcount,
    ROUND(100.0 * SUM(AttritionFlag) / COUNT(*), 1) AS attrition_rate_pct
FROM employees
GROUP BY Department, OverTime
ORDER BY Department, OverTime;
-- Finding: overtime roughly triples attrition risk in every department, but the
-- effect is strongest in Sales (13.8% -> 37.5%)

-- 4. The interaction effect: overtime x job satisfaction
SELECT
    OverTime,
    JobSatisfaction,
    COUNT(*) AS headcount,
    ROUND(100.0 * SUM(AttritionFlag) / COUNT(*), 1) AS attrition_rate_pct
FROM employees
GROUP BY OverTime, JobSatisfaction
ORDER BY OverTime, JobSatisfaction;
-- Finding: overtime + low-to-mid satisfaction clusters around 34-38% attrition,
-- vs single digits for no-overtime + high satisfaction - the two factors compound
-- rather than simply adding up, which motivated the persona segmentation approach

-- 5. Highest-risk job roles (filtered to roles with at least 20 employees,
-- to avoid small-sample noise from very small role groups)
SELECT
    JobRole,
    COUNT(*) AS headcount,
    ROUND(100.0 * SUM(AttritionFlag) / COUNT(*), 1) AS attrition_rate_pct
FROM employees
GROUP BY JobRole
HAVING COUNT(*) >= 20
ORDER BY attrition_rate_pct DESC
LIMIT 5;
-- Finding: Sales Representative (39.8%) and Laboratory Technician (23.9%) are
-- the two highest-risk roles - independently confirms the logistic regression's
-- strongest coefficients from the predictive model

-- 6. Cost of attrition by department (conditional aggregation with CASE WHEN,
-- so all departments stay in the result even if they have few/no leavers)
SELECT
    Department,
    SUM(AttritionFlag) AS leavers,
    ROUND(AVG(CASE WHEN AttritionFlag = 1 THEN EstReplacementCost END), 0) AS avg_replacement_cost,
    ROUND(SUM(CASE WHEN AttritionFlag = 1 THEN EstReplacementCost ELSE 0 END), 0) AS total_replacement_cost
FROM employees
GROUP BY Department
ORDER BY total_replacement_cost DESC;
-- Finding: Sales carries the highest total attrition cost (£6.2M) - both the
-- most leavers and an above-average replacement cost per leaver (£67,348)
