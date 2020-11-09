---- 1. create acs table with zip code data
drop table if exists cleaned.{prefix}_acs;
create table cleaned.{prefix}_acs as (
    select acs.*, link.zip
    from data_exploration.{prefix}_acs_data acs join data_exploration.acs_link al  link
    on link.tract=acs.tract and link.county = acs.county
    order by acs.tract, acs.county
);