/* Database Normalization:

1. Are the tables storing logical groupings of the data?
2. Can I make changes in a single location, rather than in
  many tables for the same information?
3. Can I access and manipulate data quickly and efficiently? */

/* JOIN and ON */
SELECT
  orders.*
FROM
  orders
JOIN
  accounts
ON
  orders.account_id = accounts.id
LIMIT
  5;

SELECT
  accounts.name,
  orders.occurred_at
FROM
  orders
JOIN
  accounts
ON
  orders.account_id = accounts.id
LIMIT
  5;

SELECT
  *
FROM
  orders
JOIN
  accounts
ON
  orders.account_id = accounts.id
LIMIT
  5;

SELECT
  *
FROM
  accounts
JOIN
  orders
ON
  accounts.id = orders.account_id;

SELECT
  orders.standard_qty,
  orders.gloss_qty,
  orders.poster_qty,
  accounts.website,
  accounts.primary_poc
FROM
  orders
JOIN
  accounts
ON
  orders.account_id = accounts.id;

/* ENTITY RELATIONSHIP DIAGRAMS (ERD)
PRIMARY KEY: unique column in a particular table
FOREIGN KEY: column in one table that is a primary key in a different table */
SELECT
  *
FROM
  web_events
JOIN
  accounts
ON
  web_events.account_id = accounts.id
JOIN
  orders
ON
  accounts.id = orders.account_id;

/* Aliases: assigned reference to an entity in the query different from its
original value, such as table1 AS t1; SELECT account AS acct */

SELECT
  web_events.*
FROM
  web_events
JOIN
  accounts
ON
  web_events.account_id = accounts.id
WHERE
  accounts.name = 'Walmart';

SELECT
  region.name Region,
  sales_reps.name Sales,
  accounts.name Acct
FROM
  region
JOIN
  sales_reps
ON
  region.id = sales_reps.region_id
JOIN
  accounts
ON
  sales_reps.id = accounts.sales_rep_id
ORDER BY
  accounts.name;

SELECT
  r.name Region_name,
  a.name Acct_name,
  (o.total_amt_usd / (o.total + 0.01)) Unit_Price
FROM
  region AS r
JOIN
  sales_reps AS s
ON
  r.id = s.region_id
JOIN
  accounts AS a
ON
  s.id = a.sales_rep_id
JOIN
  orders AS o
ON
  a.id = o.account_id;

/* LEFT and RIGHT JOIN
The table in the FROM clause is the LEFT table and
the table in the JOIN clause is the RIGHT table*/
SELECT
  L.*, R.*
FROM
  left_table l
LEFT JOIN
  right_table r
ON
  l.x = r.x

/* QUIZ */

SELECT
  r.name AS Region_name,
  s.name AS Sales_Rep_Name,
  a.name AS Account_Name
FROM
  accounts AS a
JOIN
  sales_reps AS s
ON
  a.sales_rep_id = s.id
JOIN
  region AS r
ON
  s.region_id = r.id
WHERE
  r.name = 'Midwest'
ORDER BY
  a.name;

SELECT
  r.name AS Region_name,
  s.name AS Sales_Rep_Name,
  a.name AS Account_Name
FROM
  accounts AS a
JOIN
  sales_reps AS s
ON
  a.sales_rep_id = s.id
JOIN
  region AS r
ON
  s.region_id = r.id
WHERE
  r.name = 'Midwest' AND
  s.name LIKE 'S%'
ORDER BY
  a.name;

SELECT
  r.name AS Region_name,
  s.name AS Sales_Rep_Name,
  a.name AS Account_Name
FROM
  accounts AS a
JOIN
  sales_reps AS s
ON
  a.sales_rep_id = s.id
JOIN
  region AS r
ON
  s.region_id = r.id
WHERE
  r.name = 'Midwest' AND
  s.name LIKE '% K%'
ORDER BY
  a.name;

SELECT
  r.name AS Region_name,
  a.name AS Account_Name,
  ROUND((o.total_amt_usd / (o.total + 0.01)),2) AS Unit_Price
FROM
  region AS r
JOIN
  sales_reps AS s
ON
  r.id = s.region_id
JOIN
  accounts AS a
ON
  s.id = a.sales_rep_id
JOIN
  orders AS o
ON
  a.id = o.account_id
WHERE
  o.standard_qty > 100;

SELECT
  r.name AS Region_name,
  a.name AS Account_Name,
  ROUND((o.total_amt_usd / (o.total + 0.01)),2) AS Unit_Price
FROM
  region AS r
JOIN
  sales_reps AS s
ON
  r.id = s.region_id
JOIN
  accounts AS a
ON
  s.id = a.sales_rep_id
JOIN
  orders AS o
ON
  a.id = o.account_id
WHERE
  o.standard_qty > 100 AND
  o.poster_qty > 50
ORDER BY
  Unit_Price;

SELECT
  r.name AS Region_name,
  a.name AS Account_Name,
  ROUND((o.total_amt_usd / (o.total + 0.01)),2) AS Unit_Price
FROM
  region AS r
JOIN
  sales_reps AS s
ON
  r.id = s.region_id
JOIN
  accounts AS a
ON
  s.id = a.sales_rep_id
JOIN
  orders AS o
ON
  a.id = o.account_id
WHERE
  o.standard_qty > 100 AND
  o.poster_qty > 50
ORDER BY
  Unit_Price DESC;

SELECT
  a.name AS Account_Name,
  w.channel AS Channel
FROM
  accounts AS a
JOIN
  web_events AS w
ON
  a.id = w.account_id
WHERE
  a.id = 1001
GROUP BY
  a.name,
  w.channel;

SELECT
  o.occurred_at AS Occurred_At,
  a.name AS Account_Name,
  o.total AS Total,
  o.total_amt_usd AS Total_Amt_Usd
FROM
  accounts AS a
JOIN
  orders AS o
ON
  a.id = o.account_id
WHERE
  occurred_at BETWEEN '2015-01-01' AND '2015-12-31';
