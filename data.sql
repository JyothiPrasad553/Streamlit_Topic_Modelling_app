CREATE DATABASE sentiment_app;
CREATE USER sentiment_user WITH PASSWORD 'mypassword';
GRANT ALL PRIVILEGES ON DATABASE sentiment_app TO sentiment_user;
ALTER user sentiment_user with password 'Jyothi@123'
GRANT USAGE, CREATE ON SCHEMA public TO sentiment_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO sentiment_user;