#!/usr/bin/env python3
"""
MCP Test Client for RAG Query System

This script provides an interactive command-line interface for testing
the MCP server. It handles MCP protocol formatting and displays results.
"""

import asyncio
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def run_interactive_client():
    """
    Run an interactive MCP client that communicates with the RAG server.
    """
    print("=" * 70)
    print("MCP Test Client for Allycat RAG System")
    print("=" * 70)
    print("\nConnecting to MCP server...")

    # Configure server parameters
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"],
    )

    try:
        # Connect to the server
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the session
                await session.initialize()
                print("✅ Connected to MCP server\n")

                # List available tools
                tools = await session.list_tools()
                print("Available tools:")
                for tool in tools.tools:
                    print(f"\n  Tool: {tool.name}")
                    print(f"  Description: {tool.description[:200]}...")

                print("\n" + "=" * 70)
                print("Interactive Query Mode")
                print("=" * 70)
                print("Enter your questions below. Type 'quit', 'exit', or 'q' to exit.\n")

                # Interactive query loop
                while True:
                    try:
                        # Get user input
                        user_query = input("Your question: ").strip()

                        # Check if user wants to quit
                        if user_query.lower() in ['quit', 'exit', 'q', '']:
                            if user_query == '':
                                continue
                            print("\n👋 Goodbye!")
                            break

                        print("\n⏳ Processing query...")

                        # Call the MCP tool
                        result = await session.call_tool(
                            "allycat_rag_remote",
                            arguments={"query": user_query}
                        )

                        # Display results
                        print("\n" + "-" * 70)
                        print("📋 Response:")
                        print("-" * 70)
                        for content in result.content:
                            if hasattr(content, 'text'):
                                print(content.text)
                        print("-" * 70 + "\n")

                    except KeyboardInterrupt:
                        print("\n\n👋 Goodbye!")
                        break
                    except Exception as e:
                        print(f"\n❌ Error: {e}\n")
                        continue

    except Exception as e:
        print(f"\n❌ Failed to connect to MCP server: {e}")
        print("\nMake sure:")
        print("  1. You have run 'uv sync' to install dependencies")
        print("  2. The vector database exists (run 3_save_to_vector_db.py)")
        print("  3. Your .env file is properly configured")
        sys.exit(1)


async def run_single_query(query: str):
    """
    Run a single query and exit (useful for scripting).

    Args:
        query: The question to ask
    """
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"],
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                result = await session.call_tool(
                    "allycat_rag_remote",
                    arguments={"query": query}
                )

                for content in result.content:
                    if hasattr(content, 'text'):
                        print(content.text)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """
    Main entry point - supports both interactive and single query modes.
    """
    if len(sys.argv) > 1:
        # Single query mode
        query = " ".join(sys.argv[1:])
        asyncio.run(run_single_query(query))
    else:
        # Interactive mode
        asyncio.run(run_interactive_client())


if __name__ == "__main__":
    main()
