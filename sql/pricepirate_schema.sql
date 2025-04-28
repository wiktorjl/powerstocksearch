--
-- PostgreSQL database dump
--

-- Dumped from database version 16.8 (Ubuntu 16.8-0ubuntu0.24.10.1)
-- Dumped by pg_dump version 16.8 (Ubuntu 16.8-0ubuntu0.24.10.1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: company_profile; Type: TABLE; Schema: public; Owner: pricepirate
--

CREATE TABLE public.company_profile (
    symbol_id integer,
    country character varying(2),
    currency character varying(3),
    estimate_currency character varying(3),
    exchange character varying(100),
    finnhub_industry character varying(100),
    ipo date,
    logo character varying(255),
    market_capitalization numeric(15,2),
    name character varying(100),
    phone character varying(20),
    share_outstanding numeric(10,2),
    ticker character varying(10),
    weburl character varying(255)
);


ALTER TABLE public.company_profile OWNER TO pricepirate;

--
-- Name: indicator_definitions; Type: TABLE; Schema: public; Owner: pricepirate
--

CREATE TABLE public.indicator_definitions (
    indicator_id integer NOT NULL,
    name text NOT NULL,
    description text
);


ALTER TABLE public.indicator_definitions OWNER TO pricepirate;

--
-- Name: indicator_definitions_indicator_id_seq; Type: SEQUENCE; Schema: public; Owner: pricepirate
--

CREATE SEQUENCE public.indicator_definitions_indicator_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.indicator_definitions_indicator_id_seq OWNER TO pricepirate;

--
-- Name: indicator_definitions_indicator_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: pricepirate
--

ALTER SEQUENCE public.indicator_definitions_indicator_id_seq OWNED BY public.indicator_definitions.indicator_id;


--
-- Name: indicators; Type: TABLE; Schema: public; Owner: pricepirate
--

CREATE TABLE public.indicators (
    "timestamp" date NOT NULL,
    symbol_id integer,
    indicator_id integer,
    value numeric(18,6)
);


ALTER TABLE public.indicators OWNER TO pricepirate;

--
-- Name: ohlc_data; Type: TABLE; Schema: public; Owner: pricepirate
--

CREATE TABLE public.ohlc_data (
    "timestamp" date NOT NULL,
    symbol_id integer,
    open numeric(18,6),
    high numeric(18,6),
    low numeric(18,6),
    close numeric(18,6),
    volume bigint
);


ALTER TABLE public.ohlc_data OWNER TO pricepirate;

--
-- Name: reversal_scan_results; Type: TABLE; Schema: public; Owner: pricepirate
--

CREATE TABLE public.reversal_scan_results (
    symbol text NOT NULL,
    last_close numeric(18,6),
    last_volume bigint,
    sma150 numeric(18,6),
    sma150_slope_norm numeric(18,8),
    rsi14 numeric(18,6),
    last_date date,
    scan_timestamp date NOT NULL
);


ALTER TABLE public.reversal_scan_results OWNER TO pricepirate;

--
-- Name: splits; Type: TABLE; Schema: public; Owner: pricepirate
--

CREATE TABLE public.splits (
    symbol_id integer NOT NULL,
    split_date date NOT NULL,
    ratio numeric NOT NULL
);


ALTER TABLE public.splits OWNER TO pricepirate;

--
-- Name: symbols; Type: TABLE; Schema: public; Owner: pricepirate
--

CREATE TABLE public.symbols (
    symbol_id integer NOT NULL,
    symbol text NOT NULL
);


ALTER TABLE public.symbols OWNER TO pricepirate;

--
-- Name: symbols_details; Type: TABLE; Schema: public; Owner: pricepirate
--

CREATE TABLE public.symbols_details (
    symbol_id integer NOT NULL,
    sector text NOT NULL,
    subsector text NOT NULL,
    name text NOT NULL
);


ALTER TABLE public.symbols_details OWNER TO pricepirate;

--
-- Name: symbol_info_basic; Type: VIEW; Schema: public; Owner: pricepirate
--

CREATE VIEW public.symbol_info_basic AS
 SELECT s.symbol_id,
    s.symbol,
    d.sector,
    d.subsector,
    d.name,
    p.logo,
    p.weburl,
    p.market_capitalization,
    o."timestamp",
    o.open,
    o.high,
    o.low,
    o.close,
    o.volume
   FROM (((public.symbols s
     JOIN public.symbols_details d ON ((s.symbol_id = d.symbol_id)))
     JOIN public.ohlc_data o ON ((o.symbol_id = s.symbol_id)))
     JOIN public.company_profile p ON ((p.symbol_id = s.symbol_id)));


ALTER VIEW public.symbol_info_basic OWNER TO pricepirate;

-- public.v_high_250 source

CREATE OR REPLACE VIEW public.v_high_250
AS SELECT id.name AS indicator_name,
    od."timestamp",
    s.symbol_id,
    s.symbol as symbol,
    sd.name AS symbol_name,
    od.close,
    i.value
   FROM ohlc_data od
     JOIN symbols s ON s.symbol_id = od.symbol_id
     JOIN symbols_details sd ON s.symbol_id = sd.symbol_id
     JOIN indicators i ON i.symbol_id = od.symbol_id AND i."timestamp" = od."timestamp" 
     JOIN indicator_definitions id ON id.indicator_id = i.indicator_id and id.name = 'HIGH_250';

ALTER VIEW public.v_high_250 OWNER TO pricepirate;

CREATE OR REPLACE VIEW public.v_low_250
AS SELECT id.name AS indicator_name,
    od."timestamp",
    s.symbol_id,
    s.symbol as symbol,
    sd.name AS symbol_name,
    od.close,
    i.value
   FROM ohlc_data od
     JOIN symbols s ON s.symbol_id = od.symbol_id
     JOIN symbols_details sd ON s.symbol_id = sd.symbol_id
     JOIN indicators i ON i.symbol_id = od.symbol_id AND i."timestamp" = od."timestamp"
     JOIN indicator_definitions id ON id.indicator_id = i.indicator_id AND id.name = 'LOW_250';

ALTER VIEW public.v_low_250 OWNER TO pricepirate;

CREATE OR REPLACE VIEW public.v_xabove_150
AS WITH all_indicator_data AS (
         SELECT s.symbol_id,
            s.symbol,
            sd.name AS company_name,
            sd.sector,
            sd.subsector,
            i."timestamp",
            i.value,
            lag(i.value) OVER (PARTITION BY s.symbol_id ORDER BY i."timestamp") AS previous_value,
                CASE
                    WHEN lag(i.value) OVER (PARTITION BY s.symbol_id ORDER BY i."timestamp") IS NOT NULL AND i.value > lag(i.value) OVER (PARTITION BY s.symbol_id ORDER BY i."timestamp") THEN 1
                    ELSE 0
                END AS is_increase
           FROM symbols s
             JOIN symbols_details sd ON s.symbol_id = sd.symbol_id
             JOIN indicators i ON i.symbol_id = s.symbol_id
             JOIN indicator_definitions id ON id.indicator_id = i.indicator_id
          WHERE id.name = 'SMA_150'::text
        )
 SELECT symbol,
    "timestamp" AS increase_date,
    value AS current_value,
    previous_value
   FROM all_indicator_data
  WHERE is_increase = 1
  ORDER BY "timestamp";

ALTER VIEW public.v_xabove_150 OWNER TO pricepirate;

CREATE OR REPLACE VIEW public.v_xbelow_150
AS WITH all_indicator_data AS (
         SELECT s.symbol_id,
            s.symbol,
            sd.name AS company_name,
            sd.sector,
            sd.subsector,
            i."timestamp",
            i.value,
            lag(i.value) OVER (PARTITION BY s.symbol_id ORDER BY i."timestamp") AS previous_value,
                CASE
                    WHEN lag(i.value) OVER (PARTITION BY s.symbol_id ORDER BY i."timestamp") IS NOT NULL AND i.value < lag(i.value) OVER (PARTITION BY s.symbol_id ORDER BY i."timestamp") THEN 1
                    ELSE 0
                END AS is_decrease
           FROM symbols s
             JOIN symbols_details sd ON s.symbol_id = sd.symbol_id
             JOIN indicators i ON i.symbol_id = s.symbol_id
             JOIN indicator_definitions id ON id.indicator_id = i.indicator_id
          WHERE id.name = 'SMA_150'::text
        )
 SELECT symbol,
    "timestamp" AS decrease_date,
    value AS current_value,
    previous_value
   FROM all_indicator_data
  WHERE is_decrease = 1
  ORDER BY "timestamp";

ALTER VIEW public.v_xbelow_150 OWNER TO pricepirate;


CREATE OR REPLACE VIEW public.v_strategy_highlow_250
AS SELECT vh.close AS "PRICE",
    vh."timestamp" AS "DATE",
    'OPEN'::text AS "ACTION"
   FROM v_high_250 vh
  WHERE abs(vh.close - vh.value) < 1::numeric
UNION
 SELECT vl.close AS "PRICE",
    vl."timestamp" AS "DATE",
    'CLOSE'::text AS "ACTION"
   FROM v_low_250 vl
  WHERE abs(vl.close - vl.value) < 1::numeric
  ORDER BY 2;

ALTER VIEW public.v_strategy_highlow_250 OWNER TO pricepirate;

CREATE OR REPLACE VIEW public.v_strategy_sma_150
AS SELECT "PRICE",
    "DATE",
    "SYMBOL",
    "ACTION"
   FROM ( SELECT b.current_value AS "PRICE",
            b.decrease_date AS "DATE",
            b.symbol AS "SYMBOL",
            'CLOSE'::text AS "ACTION"
           FROM v_xbelow_150 b
        UNION
         SELECT a.current_value AS "PRICE",
            a.increase_date AS "DATE",
            a.symbol AS "SYMBOL",
            'OPEN'::text AS "ACTION"
           FROM v_xabove_150 a) unnamed_subquery
  ORDER BY "DATE";

ALTER VIEW public.v_strategy_sma_150 OWNER TO pricepirate;


CREATE OR REPLACE VIEW public.v_symbols_details
AS SELECT cp.ticker, sd."name", sd.sector, sd.subsector
   FROM company_profile cp
   join symbols_details sd on sd.symbol_id = cp.symbol_id;

ALTER VIEW public.v_symbols_details OWNER TO pricepirate;

--
-- Name: symbols_symbol_id_seq; Type: SEQUENCE; Schema: public; Owner: pricepirate
--

CREATE SEQUENCE public.symbols_symbol_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.symbols_symbol_id_seq OWNER TO pricepirate;

--
-- Name: symbols_symbol_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: pricepirate
--

ALTER SEQUENCE public.symbols_symbol_id_seq OWNED BY public.symbols.symbol_id;


--
-- Name: indicator_definitions indicator_id; Type: DEFAULT; Schema: public; Owner: pricepirate
--

ALTER TABLE ONLY public.indicator_definitions ALTER COLUMN indicator_id SET DEFAULT nextval('public.indicator_definitions_indicator_id_seq'::regclass);


--
-- Name: symbols symbol_id; Type: DEFAULT; Schema: public; Owner: pricepirate
--

ALTER TABLE ONLY public.symbols ALTER COLUMN symbol_id SET DEFAULT nextval('public.symbols_symbol_id_seq'::regclass);


--
-- Name: indicator_definitions indicator_definitions_name_key; Type: CONSTRAINT; Schema: public; Owner: pricepirate
--

ALTER TABLE ONLY public.indicator_definitions
    ADD CONSTRAINT indicator_definitions_name_key UNIQUE (name);


--
-- Name: indicator_definitions indicator_definitions_pkey; Type: CONSTRAINT; Schema: public; Owner: pricepirate
--

ALTER TABLE ONLY public.indicator_definitions
    ADD CONSTRAINT indicator_definitions_pkey PRIMARY KEY (indicator_id);


--
-- Name: indicators indicators_timestamp_symbol_id_indicator_id_key; Type: CONSTRAINT; Schema: public; Owner: pricepirate
--

ALTER TABLE ONLY public.indicators
    ADD CONSTRAINT indicators_timestamp_symbol_id_indicator_id_key UNIQUE ("timestamp", symbol_id, indicator_id);


--
-- Name: ohlc_data ohlc_data_timestamp_symbol_id_key; Type: CONSTRAINT; Schema: public; Owner: pricepirate
--

ALTER TABLE ONLY public.ohlc_data
    ADD CONSTRAINT ohlc_data_timestamp_symbol_id_key UNIQUE ("timestamp", symbol_id);


--
-- Name: reversal_scan_results reversal_scan_results_pkey; Type: CONSTRAINT; Schema: public; Owner: pricepirate
--

ALTER TABLE ONLY public.reversal_scan_results
    ADD CONSTRAINT reversal_scan_results_pkey PRIMARY KEY (symbol);


--
-- Name: splits splits_pkey; Type: CONSTRAINT; Schema: public; Owner: pricepirate
--

ALTER TABLE ONLY public.splits
    ADD CONSTRAINT splits_pkey PRIMARY KEY (symbol_id, split_date);


--
-- Name: symbols_details symbols_details_pkey; Type: CONSTRAINT; Schema: public; Owner: pricepirate
--

ALTER TABLE ONLY public.symbols_details
    ADD CONSTRAINT symbols_details_pkey PRIMARY KEY (symbol_id);


--
-- Name: symbols symbols_pkey; Type: CONSTRAINT; Schema: public; Owner: pricepirate
--

ALTER TABLE ONLY public.symbols
    ADD CONSTRAINT symbols_pkey PRIMARY KEY (symbol_id);


--
-- Name: symbols symbols_symbol_key; Type: CONSTRAINT; Schema: public; Owner: pricepirate
--

ALTER TABLE ONLY public.symbols
    ADD CONSTRAINT symbols_symbol_key UNIQUE (symbol);


--
-- Name: company_profile company_profile_symbol_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: pricepirate
--

ALTER TABLE ONLY public.company_profile
    ADD CONSTRAINT company_profile_symbol_id_fkey FOREIGN KEY (symbol_id) REFERENCES public.symbols(symbol_id);


--
-- Name: indicators indicators_indicator_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: pricepirate
--

ALTER TABLE ONLY public.indicators
    ADD CONSTRAINT indicators_indicator_id_fkey FOREIGN KEY (indicator_id) REFERENCES public.indicator_definitions(indicator_id);


--
-- Name: indicators indicators_symbol_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: pricepirate
--

ALTER TABLE ONLY public.indicators
    ADD CONSTRAINT indicators_symbol_id_fkey FOREIGN KEY (symbol_id) REFERENCES public.symbols(symbol_id);


--
-- Name: ohlc_data ohlc_data_symbol_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: pricepirate
--

ALTER TABLE ONLY public.ohlc_data
    ADD CONSTRAINT ohlc_data_symbol_id_fkey FOREIGN KEY (symbol_id) REFERENCES public.symbols(symbol_id);


--
-- Name: reversal_scan_results reversal_scan_results_symbol_fkey; Type: FK CONSTRAINT; Schema: public; Owner: pricepirate
--

ALTER TABLE ONLY public.reversal_scan_results
    ADD CONSTRAINT reversal_scan_results_symbol_fkey FOREIGN KEY (symbol) REFERENCES public.symbols(symbol);


--
-- PostgreSQL database dump complete
--

