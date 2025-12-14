-- DuckDB warehouse schema (dimensional model)

create table if not exists dim_date (
    date_key integer primary key,
    date date not null,
    year integer not null,
    month integer not null,
    day integer not null
);

create table if not exists dim_symbol (
    symbol_key bigint primary key,
    symbol varchar not null unique
);

create table if not exists dim_source (
    source_key bigint primary key,
    source varchar not null unique
);

create table if not exists dim_customer (
    customer_key varchar primary key,
    email varchar,
    created_at timestamp
);

create table if not exists fact_market_prices (
    symbol_key bigint not null,
    source_key bigint not null,
    date_key integer not null,
    ts timestamp not null,
    open double,
    high double,
    low double,
    close double not null,
    volume double,
    ingested_at timestamp default now(),
    primary key (symbol_key, source_key, ts)
);

create table if not exists fact_competitive_intel (
    source_key bigint not null,
    date_key integer not null,
    ts timestamp not null,
    topic varchar,
    headline varchar,
    url varchar,
    ingested_at timestamp default now()
);

create index if not exists idx_market_prices_ts on fact_market_prices(ts);
