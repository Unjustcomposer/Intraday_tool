"""
Knowledge Graph for Indian Equity Supply Chain and Sentiment Propagation.
"""
import networkx as nx
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class EquityKnowledgeGraph:
    def __init__(self):
        """Initialize the knowledge graph."""
        self.graph = nx.DiGraph()

    def add_entity(self, ticker: str, sector: str = None, base_sentiment: float = 0.0):
        """
        Add a company entity to the graph.
        
        Args:
            ticker: The stock ticker (e.g., 'RELIANCE.NS')
            sector: The industry sector.
            base_sentiment: Initial sentiment score.
        """
        self.graph.add_node(ticker, sector=sector, sentiment=base_sentiment, shock=0.0)
        logger.debug(f"Added entity: {ticker}")

    def add_relationship(self, source_ticker: str, target_ticker: str, relationship_type: str, weight: float = 1.0):
        """
        Add a relationship between two entities.
        
        Args:
            source_ticker: Source node.
            target_ticker: Target node.
            relationship_type: Type of relation (e.g., 'SUPPLIER', 'COMPETITOR', 'CUSTOMER')
            weight: The strength of the relationship (0.0 to 1.0).
        """
        if source_ticker not in self.graph.nodes:
            self.add_entity(source_ticker)
        if target_ticker not in self.graph.nodes:
            self.add_entity(target_ticker)
            
        self.graph.add_edge(source_ticker, target_ticker, relation=relationship_type, weight=weight)
        logger.debug(f"Added relationship: {source_ticker} -> {target_ticker} ({relationship_type})")

    def apply_sentiment_shock(self, ticker: str, shock_value: float):
        """
        Apply a direct sentiment shock to a specific ticker.
        
        Args:
            ticker: The ticker receiving the shock.
            shock_value: The value of the shock (can be negative or positive).
        """
        if ticker in self.graph.nodes:
            self.graph.nodes[ticker]['shock'] += shock_value
            logger.info(f"Applied shock {shock_value} to {ticker}")
        else:
            logger.warning(f"Ticker {ticker} not in graph.")

    def propagate_shocks(self, decay_factor: float = 0.5, max_steps: int = 3) -> Dict[str, float]:
        """
        Propagate sentiment shocks through the supply chain graph.
        
        Args:
            decay_factor: How much the shock reduces at each hop.
            max_steps: Maximum number of propagation hops.
            
        Returns:
            Dictionary of updated total sentiments for each ticker.
        """
        # Initialize propagation queues
        # format: (current_node, current_shock_value, steps_remaining)
        queue = []
        for node, data in self.graph.nodes(data=True):
            if data.get('shock', 0.0) != 0.0:
                queue.append((node, data['shock'], max_steps))
        
        # Dictionary to accumulate propagated shocks
        propagated_shocks = {node: 0.0 for node in self.graph.nodes}
        
        while queue:
            current_node, shock_val, steps = queue.pop(0)
            propagated_shocks[current_node] += shock_val
            
            if steps > 0:
                for successor in self.graph.successors(current_node):
                    edge_data = self.graph.get_edge_data(current_node, successor)
                    weight = edge_data.get('weight', 1.0)
                    relation = edge_data.get('relation', 'UNKNOWN')
                    
                    # Logic to handle different relations (e.g. competitors might have inverse sentiment correlation)
                    relation_multiplier = 1.0
                    if relation == 'COMPETITOR':
                        relation_multiplier = -0.5
                    elif relation == 'SUPPLIER':
                        relation_multiplier = 0.8
                    elif relation == 'CUSTOMER':
                        relation_multiplier = 0.8
                        
                    next_shock = shock_val * weight * decay_factor * relation_multiplier
                    
                    # Only propagate if shock is significant
                    if abs(next_shock) > 0.01:
                        queue.append((successor, next_shock, steps - 1))
                        
        # Update graph sentiments and clear initial shocks
        results = {}
        for node in self.graph.nodes:
            new_sentiment = self.graph.nodes[node]['sentiment'] + propagated_shocks[node]
            self.graph.nodes[node]['sentiment'] = new_sentiment
            self.graph.nodes[node]['shock'] = 0.0  # Reset shock
            results[node] = new_sentiment
            
        return results

    def get_subgraph_for_ticker(self, ticker: str, depth: int = 2) -> nx.DiGraph:
        """
        Get the ego graph for a specific ticker up to a certain depth.
        """
        if ticker not in self.graph:
            return nx.DiGraph()
        return nx.ego_graph(self.graph, ticker, radius=depth)
