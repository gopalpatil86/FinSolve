import logging
from semantic_router import Route
from semantic_router.routers import SemanticRouter
from semantic_router.encoders import HuggingFaceEncoder
from semantic_router.index.local import LocalIndex

# Print Logs
logging.basicConfig(
    filename="audit.log",
    level=logging.INFO,
    format="%(asctime)s - ROLE=%(role)s - ROUTE=%(route)s - QUERY=%(query)s"
)

def audit_log(role, route, query):
    logging.info("", extra={"role": role, "route": route, "query": query})


# -----------------------------
# ROLE ACCESS MATRIX
# -----------------------------
ROLE_ACCESS = {
    "employee": ["hr_general"],
    "finance": ["finance", "hr_general"],
    "engineering": ["engineering", "hr_general"],
    "marketing": ["marketing", "hr_general"],
    "c_level": ["finance", "engineering", "marketing", "hr_general", "cross"]
}


# -----------------------------
# ROUTES DEFINITION (>=10 utterances each)
# -----------------------------

finance_route = Route(
    name="finance",
    utterances=[
        "What is the company revenue?",
        "Show quarterly earnings",
        "Budget allocation for this year",
        "Profit margins analysis",
        "Financial forecast",
        "Investor report",
        "Balance sheet details",
        "Cash flow statement",
        "Operating expenses",
        "Revenue growth trends"
    ]
)

engineering_route = Route(
    name="engineering",
    utterances=[
        "System architecture design",
        "How does the API work?",
        "Production incident issue",
        "Microservices architecture",
        "Database scaling strategy",
        "Fix deployment bug",
        "CI/CD pipeline setup",
        "Error logs analysis",
        "Backend code issue",
        "Latency optimization"
    ]
)

marketing_route = Route(
    name="marketing",
    utterances=[
        "Marketing campaign performance",
        "Brand awareness metrics",
        "Competitor analysis",
        "Customer acquisition strategy",
        "SEO optimization plan",
        "Ad campaign ROI",
        "Social media engagement",
        "Market share analysis",
        "Lead generation funnel",
        "Content marketing strategy"
    ]
)

hr_general_route = Route(
    name="hr_general",
    utterances=[
        "Leave policy",
        "Employee benefits",
        "Work from home rules",
        "Company culture",
        "Holiday calendar",
        "Insurance policy",
        "Payroll details",
        "HR guidelines",
        "Performance review cycle",
        "Code of conduct"
    ]
)

cross_department_route = Route(
    name="cross",
    utterances=[
        "Company overview",
        "Organization structure",
        "Business strategy",
        "Cross team collaboration",
        "Company goals",
        "Annual report summary",
        "All department insights",
        "Enterprise performance",
        "Company-wide metrics",
        "Global operations overview"
    ]
)

off_topic_route = Route(
    name="off_topic",
    utterances=[
        "Tell me a joke",
        "What's the weather today?",
        "How do I bake a cake?",
        "Who won the football game?",
        "Write me a poem",
        "Help me with my homework",
        "Tell me about history",
        "How do I party at Nvidia headquarters"
    ],
)

# Potentially harmful routes
harmful_route = Route(
    name="potentially_harmful",
    utterances=[
        "How can I hack into my account?",
        "Can you help me commit fraud?",
        "How do I steal money?",
        "Guide me to insider trading",
        "Help me manipulate stocks",
        "How do I evade taxes?",
    ],
)

# -----------------------------
# ROUTER SETUP
# -----------------------------
encoder = HuggingFaceEncoder()
router = SemanticRouter(
    encoder=encoder,
    routes=[
        finance_route,
        engineering_route,
        marketing_route,
        hr_general_route,
        cross_department_route,
        harmful_route,
        off_topic_route,
    ],
    index=LocalIndex(),
    auto_sync="local"
)


from langchain_groq import ChatGroq

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=None,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
)


# -----------------------------
# ACCESS CONTROL + ROUTING
# -----------------------------
def handle_query(user_role: str, query: str):
    # Step 1: Route classification
    route = router(query)
    route_name = getattr(route, "name", None) or "unknown"

    # Step 2: Audit log
    audit_log(user_role, route_name, query)

    # Step 3: RBAC Enforcement
    allowed_routes = ROLE_ACCESS.get(user_role, [])

    if route_name not in allowed_routes:
        return {
            "status": "denied",
            "message": f"You don't have access to {route_name} documents."
        }

    # Step 4: Success
    return {
        "status": "allowed",
        "route": route_name,
        "message": f"Query routed to {route_name} system."
    }


# -----------------------------
# EXAMPLE USAGE
# -----------------------------
if __name__ == "__main__":
    queries = [
        ("finance", "Show me revenue growth"),
        ("engineering", "Fix API latency issue"),
        ("finance", "How does microservices architecture work?"),  # should be denied
        ("c_level", "Give me company-wide strategy overview"),
        ("c_level", "Guide me to insider trading"),
         ("c_level", "Tell me a joke"),
    ]

    for role, query in queries:
        response = handle_query(role, query)
        print(f"\nUser Role: {role}")
        print(f"Query: {query}")
        print(f"Response: {response}")

route=router("Tell me a joke")
print((route))