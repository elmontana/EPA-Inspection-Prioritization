---- 1. create table with facility information
drop table if exists cleaned.{prefix}_rcra_facilities;
create table cleaned.{prefix}_rcra_facilities as (
    select
    id_number,
    lower(facility_name) as facility_name,
    lower(street_address) as street_address,
    lower(city_name) as city_name,
    substr(zip_code, 1, 5) as zip_code,
    case when trim(fed_waste_generator) = 'N' then 0 else fed_waste_generator::int end as fed_waste_generator
    from rcra.facilities
    where activity_location = 'NY' and active_site != '-----'
);

---- 2. join latitude longitude data to hhandler
drop table if exists cleaned.{prefix}_rcra_handler;
create table cleaned.{prefix}_rcra_handler as (
    select
    h.epa_handler_id, h.source_type, h.handler_sequence_number, h.receive_date, h.location_city, h.location_zip_code, h.county_code,
    h.land_type, h.importer_activity, h.mixed_waste_generator, h.transporter_activity, h.transfer_facility, h.tsd_activity,
    h.recycler_activity, h.onsite_burner_exemption, h.furnace_exemption, h.underground_injection_activity, h.receives_waste_from_off_site,
    h.universal_waste_destination_facility, h.used_oil_transporter, h.used_oil_transfer_facility, h.used_oil_processor, h.used_oil_refiner,
    h.used_oil_fuel_burner, h.used_oil_fuel_marketer_to_burner, h.used_oil_specification_marketer,
    h.under_40_cfr_part_262_subpart_k_as_a_college_or_university as subpart_k_college,
    h.under_40_cfr_part_262_subpart_k_as_a_teaching_hospital as subpart_k_hospital,
    h.under_40_cfr_part_262_subpart_k_as_a_non_profit_research_ins as subpart_k_research
    from (
        select * from rcra.hhandler where activity_location = 'NY'
    ) as h
    inner join cleaned.{prefix}_rcra_facilities as f on h.epa_handler_id = f.id_number
);

---- 3. create filtered cmecomp3 table with facility data
drop table if exists cleaned.{prefix}_rcra_cmecomp3;
create table cleaned.{prefix}_rcra_cmecomp3 as (
    select distinct on (h.epa_handler_id, c.evaluation_start_date)
    h.*,
    c.evaluation_identifier, c.evaluation_start_date, c.evaluation_agency, c.found_violation_flag,
    c.multimedia_inspection_flag, c.sampling_flag, c.not_subtitle_c_flag, c.evaluation_type, c.evaluation_type_description,
    c.focus_area, c.focus_area_description, c.evaluation_responsible_person, c.evaluation_suborganization, c.handler_activity_location,
    c.date_of_request, c.date_response_received, c.request_agency, c.request_activity_location, c.violation_activity_location,
    c.violation_sequence_number, c.violation_determined_by_agency, c.violation_type, c.violation_short_description, c.former_citation,
    c.violation_determined_date, c.actual_return_to_compliance_date, c.return_to_compliance_qualifier, c.violation_responsible_agency,
    c.scheduled_compliance_date, c.enforcement_activity_location, c.enforcement_identifier, c.enforcement_action_date, c.enforcement_agency,
    c.corrective_action_component_flag, c.appeal_initiated_date, c.appeal_resolved_date, c.disposition_status_date, c.disposition_status,
    c.disposition_status_description, c.cafo_sequence_number, c.respondent_name, c.lead_agency, c.enforcement_type, c.expenditure_amount,
    c.sep_scheduled_completion_date, c.sep_actual_completion_date, c.sep_defaulted_date, c.sep_type, c.proposed_penalty_amount,
    c.final_monetary_amount, c.paid_amount, c.final_count, c.final_amount
    from rcra.cmecomp3 c inner join cleaned.{prefix}_rcra_handler h on c.handler_id = h.epa_handler_id
    where c.evaluation_start_date >= h.receive_date
    order by h.epa_handler_id, c.evaluation_start_date, h.receive_date desc
);

---- 4. create br_reporting table
drop table if exists cleaned.{prefix}_rcra_reporting cascade; 
create table cleaned.{prefix}_rcra_reporting as (
    select r.handler_id, r.sequence_number, r.hazardous_waste_page_number, r.hazardous_waste_sub_page_number, r.br_form, r.management_location,
    r.report_cycle, r.source_code, r.form_code, r.management_method, r.federal_waste_flag, r.wastewater_characteristic_indicator,
    r.generation_tons, r.received_tons, r.waste_minimization_code, r.waste_code_group, lower(r.waste_description) as waste_description
    from rcra.br_reporting r
    inner join cleaned.{prefix}_rcra_facilities f
    on r.handler_id = f.id_number
);

---- 5. create entity id table
drop table if exists cleaned.{prefix}_entity_id;
create table if not exists cleaned.{prefix}_entity_id as(
    select distinct id_number
    from cleaned.{prefix}_rcra_facilities
    order by id_number
);
alter table cleaned.{prefix}_entity_id add column id int generated always as identity primary key;

---- 6. create event id table
drop table if exists cleaned.{prefix}_event_id;
create table if not exists cleaned.{prefix}_event_id as(
    select distinct epa_handler_id, receive_date
    from cleaned.{prefix}_rcra_cmecomp3
    order by epa_handler_id, receive_date
);
alter table cleaned.{prefix}_event_id add column id int generated always as identity primary key;
