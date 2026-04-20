"""
RONIN Digital Research – RAG-driven Personalization Engine
Generates tailored outreach emails for research respondents using
retrieval-augmented generation (FAISS + Sentence-Transformers).
"""

import os
import sys
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ------------------------------------------------------------------
# Knowledge-base artefacts (auto-created if missing)
# ------------------------------------------------------------------
KNOWLEDGE_BASE = {
    "healthcare_digital_transformation.md": """# Healthcare Digital Transformation Research 2024

## Research Background
This study explores the current state of digital transformation in healthcare institutions across Europe and Asia-Pacific. We examine how hospitals, clinics, and medical groups are adopting new technologies to improve patient outcomes, operational efficiency, and regulatory compliance.

## Target Respondents
- Hospital Chief Information Officers (CIOs)
- Medical group technology directors
- Heads of digital health initiatives

## Interview Format
45-minute in-depth interview conducted via telephone or secure video conference. All responses are anonymized and aggregated.

## Incentives
Participants receive a customized report titled "2024 Healthcare Digital Maturity Whitepaper" plus priority access to proprietary industry benchmark data.

## Key Discussion Topics
1. AI-assisted diagnosis and clinical decision support systems
2. Electronic health record (EHR) interoperability and data standards
3. Patient data privacy compliance (GDPR, HIPAA, local regulations)
4. Telemedicine infrastructure and remote patient monitoring
5. Workforce digital literacy and change management

## Why It Matters
Healthcare systems face mounting pressure to deliver better care with constrained budgets. Digital transformation is no longer optional—it is a strategic imperative for survival and growth.
""",
    "fintech_ai_adoption.md": """# AI Adoption in Global Banking Survey

## Research Background
This survey investigates how multinational banks are deploying artificial intelligence across risk management, customer service, and investment research. We map the maturity curve from early experimentation to production-scale deployment.

## Target Respondents
- Chief Technology Officers (CTOs) of retail and investment banks
- Chief Risk Officers (CROs)
- Heads of digital transformation and innovation

## Interview Format
30-minute online survey followed by an optional 20-minute supplementary deep-dive interview.

## Incentives
Respondents gain exclusive early access to the "Global Fintech AI Readiness Index" ranking. Top participants are invited to a closed-door roundtable event in Singapore.

## Key Discussion Topics
1. Generative AI compliance frameworks and regulatory sandboxes
2. Model risk management (MRM) and algorithmic auditing
3. Legacy system integration and API-first architecture
4. Customer-facing AI (chatbots, robo-advisors, personalization engines)
5. Data governance and ethical AI committees

## Why It Matters
Banks that harness AI responsibly will outperform competitors in cost efficiency, customer satisfaction, and regulatory resilience. Those that lag risk obsolescence.
""",
    "enterprise_cybersecurity.md": """# C-Suite Cybersecurity Resilience Study

## Research Background
This study evaluates how large enterprises defend against ransomware, supply-chain attacks, and insider threats. We measure preparedness across people, process, and technology dimensions.

## Target Respondents
- Chief Information Security Officers (CISOs)
- IT directors with security oversight
- Vice Presidents of risk management

## Interview Format
60-minute confidential in-depth interview. Respondents may remain completely anonymous; no attribution to individual or company is made without explicit written consent.

## Incentives
Every qualified participant receives a complimentary third-party security audit service valued at USD 5,000, plus a customized security maturity assessment report with peer benchmarking.

## Key Discussion Topics
1. Zero-trust architecture implementation and micro-segmentation
2. Cloud security posture management (CSPM) and multi-cloud governance
3. Third-party and vendor risk management (TPRM)
4. Incident response planning, tabletop exercises, and cyber insurance
5. Board-level cybersecurity literacy and budget allocation

## Why It Matters
A single successful ransomware attack can erase millions in shareholder value and destroy customer trust. Proactive resilience is the only sustainable defense.
""",
}


class PersonalizationEngine:
    """RAG-powered email personalization for RONIN research outreach."""

    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.index: faiss.Index | None = None
        self.chunks: list[str] = []
        self.chunk_metadata: list[dict] = []
        self._prompt = self._build_prompt_template()

    # ------------------------------------------------------------------
    # 1. Knowledge-base helpers
    # ------------------------------------------------------------------
    def _get_project_root(self) -> str:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _ensure_knowledge_base(self) -> str:
        """Create knowledge-base files if they do not exist."""
        kb_dir = os.path.join(self._get_project_root(), "data", "knowledge_base")
        os.makedirs(kb_dir, exist_ok=True)
        for fname, content in KNOWLEDGE_BASE.items():
            fpath = os.path.join(kb_dir, fname)
            if not os.path.exists(fpath):
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(content)
        return kb_dir

    def load_knowledge_base(self) -> list[dict]:
        """Load markdown documents from disk."""
        kb_dir = self._ensure_knowledge_base()
        docs = []
        for fname in os.listdir(kb_dir):
            if fname.endswith(".md"):
                fpath = os.path.join(kb_dir, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
                docs.append({
                    "source": fname,
                    "content": content,
                    "industry": self._map_doc_to_industry(fname),
                })
        if not docs:
            raise FileNotFoundError(f"No markdown files found in {kb_dir}")
        return docs

    @staticmethod
    def _map_doc_to_industry(fname: str) -> str:
        mapping = {
            "healthcare_digital_transformation.md": "Healthcare",
            "fintech_ai_adoption.md": "Finance",
            "enterprise_cybersecurity.md": "Technology",
        }
        return mapping.get(fname, "General")

    # ------------------------------------------------------------------
    # 2. Vector store (FAISS + Sentence-Transformers)
    # ------------------------------------------------------------------
    def build_vector_store(self, docs: list[dict]) -> None:
        """Chunk documents, embed, and build FAISS index."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        all_chunks: list[str] = []
        all_metadata: list[dict] = []

        for doc in docs:
            chunks = splitter.split_text(doc["content"])
            for chunk in chunks:
                all_chunks.append(chunk)
                all_metadata.append({
                    "source": doc["source"],
                    "industry": doc["industry"],
                })

        if not all_chunks:
            raise ValueError("No text chunks produced from knowledge base.")

        print(f"Embedding {len(all_chunks)} chunks ...")
        embeddings = self.embedding_model.encode(
            all_chunks, show_progress_bar=False, convert_to_numpy=True
        )
        embeddings = embeddings.astype("float32")
        faiss.normalize_L2(embeddings)  # cosine similarity via inner product

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        self.chunks = all_chunks
        self.chunk_metadata = all_metadata

        # Persist (use numpy+json to avoid FAISS C++ path issues on Windows)
        index_dir = os.path.join(self._get_project_root(), "models", "faiss_index_personalization")
        os.makedirs(index_dir, exist_ok=True)
        np.save(os.path.join(index_dir, "embeddings.npy"), embeddings)
        with open(os.path.join(index_dir, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump({"chunks": all_chunks, "metadata": all_metadata}, f, ensure_ascii=False)

        print(f"Vector store saved to {index_dir} ({len(all_chunks)} chunks).")

    def load_vector_store(self) -> bool:
        """Load a previously built FAISS index if available."""
        index_dir = os.path.join(self._get_project_root(), "models", "faiss_index_personalization")
        emb_path = os.path.join(index_dir, "embeddings.npy")
        chunks_path = os.path.join(index_dir, "chunks.json")

        if os.path.exists(emb_path) and os.path.exists(chunks_path):
            embeddings = np.load(emb_path).astype("float32")
            with open(chunks_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.chunks = data["chunks"]
            self.chunk_metadata = data["metadata"]

            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(embeddings)
            return True
        return False

    # ------------------------------------------------------------------
    # 3. Retrieval
    # ------------------------------------------------------------------
    def retrieve_context(
        self,
        industry: str,
        job_level: str,
        research_topic_hint: str,
        top_k: int = 3,
    ) -> list[dict]:
        """Retrieve top-k relevant chunks for a respondent profile."""
        if self.index is None:
            raise RuntimeError("Vector store not built or loaded.")

        query = f"{industry} {job_level} {research_topic_hint}"
        q_vec = self.embedding_model.encode([query], show_progress_bar=False, convert_to_numpy=True)
        q_vec = q_vec.astype("float32")
        faiss.normalize_L2(q_vec)

        # Over-fetch to allow industry filtering
        scores, indices = self.index.search(q_vec, top_k * 5)

        results: list[dict] = []
        seen: set[int] = set()
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks) or idx in seen:
                continue
            meta = self.chunk_metadata[idx]
            # Prioritize industry-matched chunks
            if meta["industry"] == industry or len(results) < top_k:
                results.append({
                    "text": self.chunks[idx],
                    "source": meta["source"],
                    "industry": meta["industry"],
                    "score": float(score),
                })
                seen.add(idx)
            if len(results) >= top_k:
                break

        return results

    # ------------------------------------------------------------------
    # 4. Email generation (template-based, RAG-augmented)
    # ------------------------------------------------------------------
    def _build_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            input_variables=[
                "salutation",
                "opening",
                "exclusivity",
                "value_prop",
                "time_commitment",
                "cta",
                "closing",
            ],
            template=(
                "{salutation},\n\n"
                "{opening}\n\n"
                "{exclusivity}{value_prop}\n\n"
                "{time_commitment}\n\n"
                "{cta}\n\n"
                "{closing}"
            ),
        )

    @staticmethod
    def _salutation(job_level: str) -> str:
        return {
            "C-Suite": "Dear Dr./Mr./Ms. [Name]",
            "Director": "Dear [Name]",
            "Manager": "Hi [Name]",
            "Specialist": "Hi [Name]",
        }.get(job_level, "Hi [Name]")

    @staticmethod
    def _industry_pain_point(industry: str) -> str:
        return {
            "Healthcare": (
                "the accelerating need for AI-driven diagnostics, interoperable health records, "
                "and patient-data privacy compliance"
            ),
            "Finance": (
                "the rapid evolution of AI in risk management, generative-AI compliance, "
                "and the challenge of modernizing legacy systems"
            ),
            "Technology": (
                "the growing threat of ransomware, the complexity of zero-trust rollouts, "
                "and securing cloud-native supply chains"
            ),
        }.get(industry, "the latest strategic developments in your industry")

    @staticmethod
    def _industry_incentive(industry: str) -> str:
        return {
            "Healthcare": (
                "a customized '2024 Healthcare Digital Maturity Whitepaper' "
                "plus priority access to industry benchmark data"
            ),
            "Finance": (
                "exclusive early access to the 'Global Fintech AI Readiness Index' "
                "and an invitation to a closed-door roundtable in Singapore"
            ),
            "Technology": (
                "a complimentary third-party security audit (valued at $5,000) "
                "plus a customized security maturity assessment report"
            ),
        }.get(industry, "exclusive research findings")

    @staticmethod
    def _base_time_commitment(industry: str) -> str:
        return {
            "Healthcare": "45 minutes",
            "Finance": "30 minutes (plus an optional 20-minute follow-up)",
            "Technology": "60 minutes",
        }.get(industry, "30 minutes")

    def generate_email(self, profile: dict) -> dict:
        """
        Generate a personalized outreach email for a single respondent.
        
        Parameters
        ----------
        profile : dict
            Must contain keys: industry, job_level, preferred_contact,
            is_hard_to_reach, past_participation_count, research_topic_match_score.
        """
        industry = profile.get("industry", "Technology")
        job_level = profile.get("job_level", "Manager")
        preferred_contact = profile.get("preferred_contact", "Email")
        is_htr = profile.get("is_hard_to_reach", 0) == 1
        past_part = profile.get("past_participation_count", 0)
        match_score = profile.get("research_topic_match_score", 50)

        # --- RAG retrieval ---
        hint_map = {
            "Healthcare": "AI-assisted diagnosis EHR interoperability patient data privacy",
            "Finance": "generative AI compliance model risk management legacy system integration",
            "Technology": "zero trust cloud security third-party risk incident response ransomware",
        }
        hint = hint_map.get(industry, "research study")
        retrieved = self.retrieve_context(industry, job_level, hint, top_k=3)

        # --- Build narrative pieces ---
        salutation = self._salutation(job_level)

        if past_part > 0:
            opening = (
                f"As a valued past participant in our research, we wanted to reach out personally. "
                f"Your profile scores {match_score}/100 on topic relevance for our upcoming study."
            )
        else:
            pain = self._industry_pain_point(industry)
            opening = (
                f"We are reaching out because your expertise aligns directly with {pain}. "
                f"Our internal relevance model scores your profile {match_score}/100 for this study."
            )

        exclusivity = ""
        if is_htr:
            exclusivity = (
                "This is a strictly by-invitation-only study limited to 50 global executives. "
            )

        incentive = self._industry_incentive(industry)
        value_prop = f"In exchange for your time, you will receive {incentive}."

        base_time = self._base_time_commitment(industry)
        if is_htr:
            time_commitment = (
                f"We value your time: a streamlined "
                f"{base_time.replace('45 minutes', '20-minute executive briefing').replace('60 minutes', '20-minute executive briefing').replace('30 minutes', '15-minute executive briefing').replace(' (plus an optional 20-minute follow-up)', '')} "
                f"is all we need."
            )
        else:
            time_commitment = f"The time commitment is {base_time}."

        # Channel adaptation
        is_linkedin = preferred_contact == "LinkedIn"
        if is_linkedin:
            cta = "Would you be open to a quick conversation? Reply here and I'll send a calendar link."
        else:
            cta = (
                "Please reply to this email or click the calendar link below "
                "to schedule a time that works best for you."
            )

        closing = "Best regards,\nThe RONIN International Research Team"

        # --- Assemble via LangChain PromptTemplate ---
        body = self._prompt.format(
            salutation=salutation,
            opening=opening,
            exclusivity=exclusivity,
            value_prop=value_prop,
            time_commitment=time_commitment,
            cta=cta,
            closing=closing,
        )

        # Inject RAG context (only for Email channel to keep LinkedIn concise)
        if not is_linkedin:
            context_block = "\n\n---\nRelevant research context:\n"
            for i, r in enumerate(retrieved, 1):
                # Truncate long chunks for readability
                snippet = r["text"][:280].replace("\n", " ")
                context_block += f"{i}. [{r['industry']}] {snippet}...\n"
            body += context_block

        # --- Length enforcement for LinkedIn ---
        if is_linkedin:
            words = body.split()
            if len(words) > 100:
                body = " ".join(words[:95]) + " …"

        subject = self._generate_subject(industry, is_htr)

        return {
            "subject": subject,
            "body": body,
            "channel": preferred_contact,
            "industry": industry,
            "job_level": job_level,
            "retrieved_context": retrieved,
        }

    @staticmethod
    def _generate_subject(industry: str, is_htr: bool) -> str:
        base = f"[Exclusive Invitation] {industry} Research Study"
        if is_htr:
            base = f"[By Invitation Only] {industry} Executive Research – 20 Min"
        return base

    # ------------------------------------------------------------------
    # 5. Relevance evaluation
    # ------------------------------------------------------------------
    def evaluate_relevance(self, email_text: str, industry: str) -> float:
        """Cosine similarity between email and industry-specific KB chunks."""
        industry_chunks = [
            c for c, m in zip(self.chunks, self.chunk_metadata)
            if m["industry"] == industry
        ]
        if not industry_chunks:
            industry_chunks = self.chunks

        email_vec = self.embedding_model.encode(
            [email_text], show_progress_bar=False, convert_to_numpy=True
        )
        chunk_vecs = self.embedding_model.encode(
            industry_chunks, show_progress_bar=False, convert_to_numpy=True
        )

        # Normalise
        email_norm = email_vec / np.linalg.norm(email_vec, axis=1, keepdims=True)
        chunk_norm = chunk_vecs / np.linalg.norm(chunk_vecs, axis=1, keepdims=True)

        sims = np.dot(chunk_norm, email_norm.T).flatten()
        return float(sims.mean()) * 100  # percentage

    # ------------------------------------------------------------------
    # 6. Persistence helpers
    # ------------------------------------------------------------------
    def save_sample_emails(self, emails: list[dict], filename: str = "sample_emails.json") -> None:
        out_path = os.path.join(self._get_project_root(), "data", filename)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(emails, f, ensure_ascii=False, indent=2)
        print(f"Sample emails saved to: {out_path}")


# =====================================================================
# Main execution block
# =====================================================================
if __name__ == "__main__":
    engine = PersonalizationEngine()

    # Build or load vector store
    if not engine.load_vector_store():
        print("Building vector store from knowledge base...")
        docs = engine.load_knowledge_base()
        engine.build_vector_store(docs)
    else:
        print(f"Loaded vector store ({len(engine.chunks)} chunks).")

    # Define 3 canonical respondent personas
    profiles = [
        {
            "id": "Respondent_A_HardToReach",
            "industry": "Healthcare",
            "job_level": "C-Suite",
            "preferred_contact": "Email",
            "is_hard_to_reach": 1,
            "past_participation_count": 0,
            "research_topic_match_score": 88,
        },
        {
            "id": "Respondent_B_ReturningCustomer",
            "industry": "Finance",
            "job_level": "Director",
            "preferred_contact": "LinkedIn",
            "is_hard_to_reach": 0,
            "past_participation_count": 2,
            "research_topic_match_score": 75,
        },
        {
            "id": "Respondent_C_NewLead",
            "industry": "Technology",
            "job_level": "Manager",
            "preferred_contact": "Email",
            "is_hard_to_reach": 0,
            "past_participation_count": 0,
            "research_topic_match_score": 62,
        },
    ]

    results: list[dict] = []
    for p in profiles:
        print(f"\n{'='*50}")
        print(f"Generating email for {p['id']}")
        print(f"Profile: {p['industry']} | {p['job_level']} | {p['preferred_contact']} | HtR={p['is_hard_to_reach']}")
        print("=" * 50)

        email = engine.generate_email(p)
        relevance = engine.evaluate_relevance(email["body"], p["industry"])

        print(f"Subject: {email['subject']}")
        print(f"Channel: {email['channel']}")
        print("-" * 50)
        print(email["body"])
        print("-" * 50)
        print(f"邮件与研究主题相关性分数：{relevance:.1f}%")

        results.append({
            "profile_id": p["id"],
            "profile": p,
            "email": email,
            "relevance_score_percent": round(relevance, 1),
        })

    engine.save_sample_emails(results)
    print("\nAll sample emails generated and saved successfully.")
