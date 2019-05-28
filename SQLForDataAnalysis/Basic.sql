/* Suggested to keep a notebook */
SELECT
	occurred_at, account_id, channel
FROM
	web_events
LIMIT 15;

/* Including Order By clause */
SELECT
	occurred_at, account_id, channel
FROM
	web_events
ORDER BY
  occurred_at
LIMIT 15;

SELECT
  id, occurred_at, total_amt_usd
FROM
  orders
ORDER BY
  occurred_at
LIMIT 10;

SELECT
	id, account_id, total_amt_usd
FROM
	orders
ORDER BY
	total_amt_usd DESC
LIMIT 5;

SELECT
	id, account_id, total_amt_usd
FROM
	orders
ORDER BY
	total_amt_usd
LIMIT 20;

/* Including multiple Order By clause */
SELECT
	id, account_id, total_amt_usd
FROM
	orders
ORDER BY
	account_id, total_amt_usd DESC;

SELECT
	id, account_id, total_amt_usd
FROM
	orders
ORDER BY
	 total_amt_usd DESC, account_id;

-- The secondary sorting by account ID is difficult
-- to see here, since only if there were two orders
-- with equal total dollar amounts would there need
-- to be any sorting by account ID.

/* Using the Where clause */
SELECT
	*
FROM
	orders
WHERE
	gloss_amt_usd >= 1000
LIMIT 5;

SELECT
	*
FROM
	orders
WHERE
	total_amt_usd < 500
ORDER BY
	total_amt_usd DESC
LIMIT 10;

/* Using WHERE with Non-Numeric Data */
SELECT
	name,
    website,
    primary_poc
FROM
	accounts
WHERE
	name = 'Exxon Mobil';

/* Derived columns */
SELECT
	id,
	(standard_amt_usd/total_amt_usd)*100 AS std_percent,
	total_amt_usd
FROM
	orders
LIMIT 10;

SELECT
	id,
	account_id,
	(standard_amt_usd/standard_qty)
FROM
	orders
LIMIT 10;

SELECT
	id,
	account_id,
	poster_amt_usd/(standard_amt_usd + gloss_amt_usd + poster_amt_usd) AS poster_rev_perc
FROM
	orders
LIMIT 10;
