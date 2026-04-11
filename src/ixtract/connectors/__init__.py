"""Data source connectors."""
from ixtract.connectors.base import BaseConnector, ObjectMetadata, LatencyProfile, SourceConnections
from ixtract.connectors.mysql import MySQLConnector

# SQL Server connector imported lazily (requires pyodbc)
# from ixtract.connectors.sqlserver import SQLServerConnector

__all__ = [
    "BaseConnector", "ObjectMetadata", "LatencyProfile", "SourceConnections",
    "MySQLConnector",
]
