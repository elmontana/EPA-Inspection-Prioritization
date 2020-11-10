-- below is just a copy paste of above, with the first two lines changed to create in semantic schema
-- perhaps something else needs to be changed?
drop table if exists semantic.{prefix}_acs;
create table semantic.{prefix}_acs as (
	select e.id as entity_id, acs_processed.zip, acs_processed.zip_population, acs_processed.county, acs_processed.county_population from
	(
	select distinct f.id_number, t.*, zip_county_pop_table.county, zip_county_pop_table.county_population from data_exploration.rcra_facilities f
	left join (
		select x.zip, sum(x."B01003_001E") zip_population -- add more sums here for more fields
		from (select l.zip, a.*
		from data_exploration.acs a
		inner join data_exploration.acs_link l
		on a.state = l.state and a.county = l.county and a.tract = l.tract) x
		group by x.zip) t
	on f.zip_code = t.zip

	left join (
		select linking_table.zip, county_pop_table.* from data_exploration.acs_link linking_table
		left join(
		select a.county, sum(a."B01003_001E") county_population -- add more sums here for more fields
		from data_exploration.acs a
		group by a.county) county_pop_table
	on linking_table.county = county_pop_table.county) zip_county_pop_table
	on f.zip_code = zip_county_pop_table.zip
	) as acs_processed
	inner join cleaned.{prefix}_entity_id e
	on acs_processed.id_number = e.id_number
);
