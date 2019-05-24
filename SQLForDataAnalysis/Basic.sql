/* Suggested to keep a notebook */
SELECT
	occurred_at, account_id, channel
FROM
	web_events
LIMIT 15;
