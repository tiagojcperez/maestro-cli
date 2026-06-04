from __future__ import annotations

from maestro_cli.knowledge_graph import (
    Entity,
    KnowledgeGraph,
    Relation,
    build_knowledge_graph,
)


# ===========================================================================
# KnowledgeGraph.get_related — reverse traversal + early frontier exhaustion
# ===========================================================================


class TestGetRelatedReverseTraversal:
    def test_reverse_edge_followed(self) -> None:
        """A relation whose target is the current node walks back to its source.

        Drives the branch that adds ``rel.source_id`` when ``rel.target_id``
        matches the frontier node (reverse-direction traversal).
        """
        g = KnowledgeGraph()
        g.add_entity(Entity(id="a", name="A", entity_type="file"))
        g.add_entity(Entity(id="b", name="B", entity_type="function"))
        # Edge points a -> b; starting from b we must reach a via the reverse arm.
        g.add_relation(Relation(source_id="a", target_id="b", relation_type="defines"))

        related = g.get_related("b", max_hops=2)

        assert "a" in related
        assert "b" in related

    def test_frontier_empties_before_max_hops(self) -> None:
        """When the frontier empties, the loop breaks early instead of spinning.

        With a single one-hop edge and a large max_hops, the second iteration
        starts with an empty frontier, hitting the early-break path.
        """
        g = KnowledgeGraph()
        g.add_entity(Entity(id="a", name="A", entity_type="file"))
        g.add_entity(Entity(id="b", name="B", entity_type="function"))
        g.add_relation(Relation(source_id="a", target_id="b", relation_type="defines"))

        related = g.get_related("a", max_hops=10)

        assert related == {"a", "b"}

    def test_isolated_node_breaks_immediately(self) -> None:
        """A node with no edges produces an empty next frontier on hop 1."""
        g = KnowledgeGraph()
        g.add_entity(Entity(id="lonely", name="L", entity_type="concept"))

        related = g.get_related("lonely", max_hops=5)

        assert related == {"lonely"}


# ===========================================================================
# KnowledgeGraph.format_context — budget breaks + properties rendering
# ===========================================================================


class TestFormatContextBudget:
    def test_type_header_exceeds_budget_breaks(self) -> None:
        """A tiny budget that cannot fit even the first type header breaks out.

        The very first ``## <Type>s`` header is longer than the budget, so the
        type-loop break fires before any entity is appended.
        """
        g = KnowledgeGraph()
        g.add_entity(Entity(id="a", name="A", entity_type="function"))

        out = g.format_context(budget_chars=1)

        assert out == ""

    def test_entity_properties_rendered(self) -> None:
        """An entity with properties renders the ' — k: v' suffix."""
        g = KnowledgeGraph()
        g.add_entity(
            Entity(
                id="a",
                name="parse",
                entity_type="function",
                source_task="impl",
                properties={"lang": "python"},
            )
        )

        out = g.format_context(budget_chars=24000)

        assert "lang: python" in out
        assert "parse" in out

    def test_entity_line_exceeds_budget_breaks(self) -> None:
        """Budget fits the header but not the entity line — inner loop breaks.

        The header ``## Functions\n`` (~13 chars) fits, but the entity line is
        longer than the remaining budget, so the entity loop breaks.
        """
        g = KnowledgeGraph()
        g.add_entity(
            Entity(
                id="a",
                name="a_very_long_function_name_that_overflows",
                entity_type="function",
                source_task="impl",
            )
        )

        # Budget large enough for the header but not for the long entity line.
        header_len = len("## Functions\n")
        out = g.format_context(budget_chars=header_len + 2)

        assert "## Functions" in out
        # The entity line was skipped because it overflowed the budget.
        assert "a_very_long_function_name_that_overflows" not in out

    def test_relation_line_exceeds_budget_breaks(self) -> None:
        """Header + entities fit, then the relationships section line overflows.

        We size the budget so all entity output and the Relationships header are
        emitted, but the first formatted relation line overflows, hitting the
        relation-loop break.
        """
        g = KnowledgeGraph()
        g.add_entity(Entity(id="a", name="A", entity_type="file"))
        g.add_entity(Entity(id="b", name="B", entity_type="file"))
        g.add_relation(Relation(source_id="a", target_id="b", relation_type="defines"))

        # First render with a generous budget to learn the prefix length up to
        # and including the Relationships header.
        full = g.format_context(budget_chars=100000)
        rel_marker = "## Relationships\n"
        prefix_len = full.index(rel_marker) + len(rel_marker)

        # Budget covers everything through the Relationships header but leaves
        # too little room for the actual relation line.
        out = g.format_context(budget_chars=prefix_len + 2)

        assert "## Relationships" in out
        # The relation arrow line must not be present (it overflowed).
        assert "—[defines]→" not in out


# ===========================================================================
# build_knowledge_graph — fallback truncation break
# ===========================================================================


class TestBuildKnowledgeGraphFallback:
    def test_fallback_break_when_header_exceeds_remaining(self) -> None:
        """No entities extracted: fallback truncates, breaking when a header
        no longer fits the remaining char budget.

        Two upstreams with content that yields no entities. A tiny budget lets
        the first ``--- uid ---`` header consume almost everything, so the
        second iteration's header no longer fits and the loop breaks.
        """
        # Plain prose with no file paths, defs, classes, decisions, errors,
        # or import keywords -> extract_entities returns nothing.
        upstream = {
            "alpha": "the quick brown fox jumps over",
            "beta": "lazy dog naps softly here today",
        }

        # budget_tokens * 4 = char budget. Choose a value where the first
        # header (14 chars) + a sliver of truncated text consumes the budget,
        # so the second iteration's header no longer fits and the loop breaks.
        out = build_knowledge_graph(upstream, budget_tokens=5)

        # Only the first upstream should appear; the second header was skipped.
        assert "--- alpha ---" in out
        assert "--- beta ---" not in out

    def test_fallback_returns_truncated_when_no_entities(self) -> None:
        """Sanity: fallback path returns content (not graph) when no entities."""
        upstream = {"only": "plain words with nothing structured at all here"}
        out = build_knowledge_graph(upstream, budget_tokens=50)
        assert "--- only ---" in out
        assert "plain words" in out
