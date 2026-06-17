from __future__ import annotations

from typing import Any

from multi_agent_sync.events.event import AgentEvent


COMMUNICATION_EVENT_TYPES = {"finding", "critique", "warning", "question"}


def build_sync_report(event_log: list[AgentEvent], agent_traces: dict[str, dict[str, Any]]) -> dict[str, Any]:
    receipts = _receipts_by_event_and_agent(agent_traces)
    sent_events = [event for event in event_log if event.event_type in COMMUNICATION_EVENT_TYPES]
    agent_pairs = []

    for event in sent_events:
        for target_agent in _target_agents(event, agent_traces):
            receipt = receipts.get((event.event_id, target_agent))
            unused_reason = _unused_reason(agent_traces.get(target_agent, {}), event.event_id)
            agent_pairs.append(
                {
                    "from": event.source,
                    "to": target_agent,
                    "event_id": event.event_id,
                    "event_type": event.event_type,
                    "received": receipt is not None,
                    "accepted": bool(receipt and receipt.get("accepted")),
                    "used": bool(receipt and receipt.get("used_in_step") is not None),
                    "used_in_step": receipt.get("used_in_step") if receipt else None,
                    "reason": _pair_reason(receipt, unused_reason),
                }
            )

    all_receipts = [receipt for trace in agent_traces.values() for receipt in trace.get("event_receipts", [])]
    accepted_receipts = [receipt for receipt in all_receipts if receipt.get("accepted")]
    used_receipts = [receipt for receipt in accepted_receipts if receipt.get("used_in_step") is not None]

    return {
        "messages_sent": len(sent_events),
        "messages_received": len(all_receipts),
        "messages_accepted": len(accepted_receipts),
        "messages_ignored": len(all_receipts) - len(accepted_receipts),
        "messages_used_in_prompt": len(used_receipts),
        "messages_received_but_unused": len(accepted_receipts) - len(used_receipts),
        "agent_pairs": agent_pairs,
    }


def _receipts_by_event_and_agent(agent_traces: dict[str, dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    receipts: dict[tuple[str, str], dict[str, Any]] = {}
    for agent_name, trace in agent_traces.items():
        for receipt in trace.get("event_receipts", []):
            receipts[(receipt["event_id"], agent_name)] = receipt
    return receipts


def _target_agents(event: AgentEvent, agent_traces: dict[str, dict[str, Any]]) -> list[str]:
    if event.target not in (None, "broadcast"):
        return [event.target]
    return [agent_name for agent_name in agent_traces if agent_name != event.source]


def _unused_reason(trace: dict[str, Any], event_id: str) -> str | None:
    for unused_event in trace.get("unused_received_events", []):
        if unused_event.get("event_id") == event_id:
            return unused_event.get("reason")
    return None


def _pair_reason(receipt: dict[str, Any] | None, unused_reason: str | None) -> str | None:
    if receipt is None:
        return "sent_but_not_received"
    if not receipt.get("accepted"):
        return receipt.get("ignored_reason")
    if receipt.get("used_in_step") is None:
        return unused_reason or "accepted_but_not_used_in_any_prompt"
    return None
