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

/* Using the WHERE clause */
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

/* LIKE */
-- LIKE function requires the use of wildcards, %, to indicate any number
-- of characters leading up to or following the wildcard
SELECT
	name
FROM
	accounts
WHERE
	name LIKE 'C%';

SELECT
	name
FROM
	accounts
WHERE
	name LIKE '%one%';

SELECT
	name
FROM
	accounts
WHERE
	name LIKE '%s';

/* IN function: filter data on several values*/
SELECT
	name,
	primary_poc,
	sales_rep_id
FROM
	accounts
WHERE
	name IN ('Walmart', 'Target', 'Nordstrom');

SELECT
	*
FROM
	web_events
WHERE
	channel IN ('organic', 'adwords');

/* NOT function: negation of a function*/
SELECT
	name,
	primary_poc,
	sales_rep_id
FROM
	accounts
WHERE
	name NOT IN ('Walmart', 'Target', 'Nordstrom');

SELECT
	*
FROM
	web_events
WHERE
	channel NOT IN ('organic', 'adwords');

	SELECT
		name
	FROM
		accounts
	WHERE
		name NOT LIKE 'C%';

	SELECT
		name
	FROM
		accounts
	WHERE
		name NOT LIKE '%one%';

	SELECT
		name
	FROM
		accounts
	WHERE
		name NOT LIKE '%s';

/* AND function: boolean for numerous conditions being pairwise (BETWEEN may
be useful as well, BETWEEN is inclusive) */
SELECT
	*
FROM
	orders
WHERE
	standard_qty > 1000 AND
	poster_qty = 0 AND
	gloss_qty = 0;

SELECT
	name
FROM
	accounts
WHERE
	name NOT LIKE 'C%' AND
	name NOT LIKE '%s';

SELECT
	occurred_at,
	gloss_qty
FROM
	orders
WHERE
	gloss_qty BETWEEN 24 AND 29
ORDER BY
	gloss_qty;

SELECT
	*
FROM
	web_events
WHERE
	channel IN ('organic', 'adwords') AND
	occurred_at BETWEEN '2016-01-01' AND '2017-01-01'
ORDER BY
	occurred_at ASC;

/* OR function: boolean test for there exists */
SELECT
	id
FROM
	orders
WHERE
	gloss_qty > 4000 OR
	poster_qty > 4000
ORDER BY
	id;

SELECT
	*
FROM
	orders
WHERE
	standard_qty = 0 AND
	(gloss_qty > 1000 OR poster_qty > 1000)
ORDER BY
	id;

SELECT
	*
FROM
	accounts
WHERE
	(name LIKE 'C%' OR name LIKE 'W%') AND
	(primary_poc LIKE '%ana%' or primary_poc LIKE '%Ana') AND
	primary_poc NOT LIKE '%eana%'
ORDER BY
	id;
