drop table if exists semantic.{prefix}_events;
create table semantic.{prefix}_events as (
    select t.id as entity_id, v.id as event_id, d.event_date, d.knowledge_date, d.found_violation, d.citizen_complaint, d.penalty_amount from (
        select
        epa_handler_id, receive_date,
        max(evaluation_start_date) event_date,
        max(greatest(evaluation_start_date, violation_determined_date, actual_return_to_compliance_date, enforcement_action_date, disposition_status_date)) as knowledge_date,
        case when (sum(case when found_violation_flag = 'Y' then 1 else 0 end) > 0) then 1 else 0 end as found_violation,
        case when (sum(case when citizen_complaint_flag = 'Y' then 1 else 0 end) > 0) then 1 else 0 end as citizen_complaint,
        coalesce(max(final_amount), 0) as penalty_amount
        from cleaned.{prefix}_rcra_cmecomp3
        group by epa_handler_id, receive_date
        order by epa_handler_id, receive_date) as d
    inner join cleaned.{prefix}_entity_id t on t.id_number = d.epa_handler_id
    inner join cleaned.{prefix}_event_id v on v.epa_handler_id = d.epa_handler_id and v.receive_date = d.receive_date
);
