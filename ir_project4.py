"""
RAG-Powered Multi-Website Job Intelligence System
Combines Indeed, LinkedIn, ZipRecruiter with semantic search and intelligent matching
"""

import os
import requests
from bs4 import BeautifulSoup
import time
import random
import json
from typing import List, Dict, Optional, Tuple, Any, Set
import pandas as pd
from datetime import datetime, timedelta
import re
import hashlib
from dataclasses import dataclass, asdict, field
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import pickle
from enum import Enum

# RAG Core Components
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    print("Note: Install RAG components: pip install sentence-transformers scikit-learn")

# Optional LLM Integration
try:
    import openai
    from openai import OpenAI
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

class JobCategory(Enum):
    """Job categories for intelligent classification"""
    TECH = "Technology"
    FINANCE = "Finance"
    HEALTHCARE = "Healthcare"
    ENGINEERING = "Engineering"
    MARKETING = "Marketing"
    SALES = "Sales"
    DESIGN = "Design"
    DATA = "Data Science"
    MANAGEMENT = "Management"
    OPERATIONS = "Operations"
    CUSTOMER_SERVICE = "Customer Service"
    EDUCATION = "Education"
    OTHER = "Other"

@dataclass
class JobProfile:
    """RAG-enhanced job profile with embeddings"""
    id: str
    title: str
    company: str
    city: str
    state: str
    full_location: str
    salary: str
    salary_min: Optional[float] = None
    salary_max: Optional[float] = None
    salary_avg: Optional[float] = None
    experience: str
    description: str
    requirements: str
    skills: List[str]
    source: str
    url: str
    posted_date: str
    job_type: str = ""
    remote_status: str = ""
    industry: str = ""
    benefits: List[str] = field(default_factory=list)
    company_size: str = ""
    job_level: str = ""
    education: str = ""

    # RAG Fields
    embedding: Optional[np.ndarray] = None
    semantic_score: float = 0.0
    category: JobCategory = JobCategory.OTHER
    keywords: List[str] = field(default_factory=list)
    summary: str = ""

    # Metadata
    relevance_tags: List[str] = field(default_factory=list)
    cluster_id: int = -1
    match_reasons: List[str] = field(default_factory=list)

    def to_dict(self):
        data = asdict(self)
        data['embedding'] = self.embedding.tolist() if self.embedding is not None else None
        data['category'] = self.category.value
        return data

    def get_text_for_embedding(self) -> str:
        """Get text for embedding generation"""
        return f"""
        Title: {self.title}
        Company: {self.company}
        Location: {self.full_location}
        Skills: {', '.join(self.skills)}
        Experience: {self.experience}
        Description: {self.description[:500]}
        Requirements: {self.requirements[:500]}
        """

class SemanticSearchEngine:
    """Vector-based semantic search engine for jobs"""

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        if EMBEDDING_AVAILABLE:
            self.model = SentenceTransformer(model_name)
        else:
            self.model = None
        self.jobs: List[JobProfile] = []
        self.embeddings: np.ndarray = None
        self.job_index: Dict[str, int] = {}
        self.keyword_index: Dict[str, Set[int]] = defaultdict(set)

        # Initialize TF-IDF for keyword search
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.tfidf_matrix = None
        self.tfidf_feature_names = None

    def add_job(self, job: JobProfile):
        """Add job to search engine with embedding"""
        idx = len(self.jobs)
        self.jobs.append(job)
        self.job_index[job.id] = idx

        # Add to keyword index
        for skill in job.skills:
            self.keyword_index[skill.lower()].add(idx)

        for keyword in job.keywords:
            self.keyword_index[keyword.lower()].add(idx)

    def build_index(self):
        """Build search indexes"""
        if not self.jobs:
            return

        # Build embeddings
        if self.model:
            texts = [job.get_text_for_embedding() for job in self.jobs]
            self.embeddings = self.model.encode(texts)

            # Add embeddings to jobs
            for i, job in enumerate(self.jobs):
                job.embedding = self.embeddings[i]

        # Build TF-IDF index
        job_texts = [
            f"{job.title} {job.company} {' '.join(job.skills)} {job.description[:200]}"
            for job in self.jobs
        ]
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(job_texts)
        self.tfidf_feature_names = self.tfidf_vectorizer.get_feature_names_out()

    def semantic_search(self, query: str, top_k: int = 10, threshold: float = 0.3) -> List[JobProfile]:
        """Semantic search using embeddings"""
        if not self.model or self.embeddings is None:
            return []

        query_embedding = self.model.encode([query])[0]
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]

        # Get top-k results
        indices = np.argsort(similarities)[::-1][:top_k * 2]  # Get more for filtering

        results = []
        for idx in indices:
            if similarities[idx] >= threshold:
                job = self.jobs[idx]
                job.semantic_score = float(similarities[idx])

                # Generate match reasons
                job.match_reasons = self._generate_match_reasons(query, job, similarities[idx])

                results.append(job)
                if len(results) >= top_k:
                    break

        return results

    def hybrid_search(self, query: str, filters: Dict = None, top_k: int = 20) -> List[JobProfile]:
        """Hybrid search combining semantic and keyword search"""
        # Semantic search
        semantic_results = self.semantic_search(query, top_k * 2)

        # Keyword search
        keyword_results = self.keyword_search(query)

        # Combine results
        job_scores = defaultdict(float)

        # Score semantic results
        for job in semantic_results:
            job_scores[job.id] += job.semantic_score * 0.7

        # Score keyword results
        for job in keyword_results:
            job_scores[job.id] += 0.3

        # Apply filters
        filtered_jobs = []
        for job_id, score in sorted(job_scores.items(), key=lambda x: x[1], reverse=True):
            job = self.jobs[self.job_index[job_id]]

            if self._apply_filters(job, filters):
                job.semantic_score = score
                filtered_jobs.append(job)

            if len(filtered_jobs) >= top_k:
                break

        return filtered_jobs

    def keyword_search(self, query: str) -> List[JobProfile]:
        """Keyword search using TF-IDF"""
        if self.tfidf_matrix is None:
            return []

        # Transform query
        query_vec = self.tfidf_vectorizer.transform([query])

        # Calculate similarities
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        # Get top results
        indices = np.argsort(similarities)[::-1][:20]

        results = []
        for idx in indices:
            if similarities[idx] > 0.1:
                job = self.jobs[idx]
                job.semantic_score = float(similarities[idx])
                results.append(job)

        return results

    def find_similar_jobs(self, job_id: str, top_k: int = 10) -> List[JobProfile]:
        """Find jobs similar to a given job"""
        if job_id not in self.job_index:
            return []

        idx = self.job_index[job_id]
        job_embedding = self.embeddings[idx]

        similarities = cosine_similarity([job_embedding], self.embeddings)[0]
        similarities[idx] = -1  # Exclude self

        indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for i in indices:
            if similarities[i] > 0.3:
                similar_job = self.jobs[i]
                similar_job.semantic_score = float(similarities[i])
                results.append(similar_job)

        return results

    def cluster_jobs(self, n_clusters: int = 10) -> Dict[int, List[JobProfile]]:
        """Cluster jobs using K-means"""
        if self.embeddings is None:
            return {}

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(self.embeddings)

        clusters = defaultdict(list)
        for job, label in zip(self.jobs, cluster_labels):
            job.cluster_id = label
            clusters[label].append(job)

        return dict(clusters)

    def _generate_match_reasons(self, query: str, job: JobProfile, score: float) -> List[str]:
        """Generate reasons why job matches query"""
        reasons = []

        # Check title match
        if any(word.lower() in job.title.lower() for word in query.split()):
            reasons.append("Job title matches your search")

        # Check skills match
        query_skills = [word.lower() for word in query.split() if len(word) > 3]
        matched_skills = [skill for skill in job.skills if any(qs in skill.lower() for qs in query_skills)]
        if matched_skills:
            reasons.append(f"Requires skills: {', '.join(matched_skills[:3])}")

        # Check location
        if 'remote' in query.lower() and job.remote_status == 'Remote':
            reasons.append("Remote position available")

        # Check experience level
        exp_keywords = ['entry', 'junior', 'mid', 'senior', 'lead', 'principal']
        for exp in exp_keywords:
            if exp in query.lower() and exp in job.experience.lower():
                reasons.append(f"{exp.title()} level position")
                break

        # Add semantic match confidence
        if score > 0.7:
            reasons.append("High semantic match with your query")
        elif score > 0.5:
            reasons.append("Good semantic match with your query")

        return reasons[:3]

    def _apply_filters(self, job: JobProfile, filters: Dict) -> bool:
        """Apply filters to job"""
        if not filters:
            return True

        # Location filter
        if 'locations' in filters:
            if job.full_location not in filters['locations'] and job.state not in filters['locations']:
                return False

        # Remote filter
        if 'remote_only' in filters and filters['remote_only']:
            if job.remote_status != 'Remote':
                return False

        # Experience filter
        if 'experience_level' in filters:
            exp_filter = filters['experience_level'].lower()
            job_exp = job.experience.lower()

            if exp_filter == 'entry' and 'senior' in job_exp:
                return False
            elif exp_filter == 'senior' and 'entry' in job_exp:
                return False

        # Salary filter
        if 'min_salary' in filters and job.salary_avg:
            if job.salary_avg < filters['min_salary']:
                return False

        return True

class RAGJobAnalyzer:
    """RAG-based job analyzer with intelligent insights"""

    def __init__(self, search_engine: SemanticSearchEngine):
        self.search_engine = search_engine
        self.job_categories = self._load_job_categories()

    def _load_job_categories(self) -> Dict[str, JobCategory]:
        """Load job category mappings"""
        return {
            'software': JobCategory.TECH,
            'developer': JobCategory.TECH,
            'engineer': JobCategory.TECH,
            'data': JobCategory.DATA,
            'analyst': JobCategory.DATA,
            'scientist': JobCategory.DATA,
            'finance': JobCategory.FINANCE,
            'accountant': JobCategory.FINANCE,
            'banking': JobCategory.FINANCE,
            'health': JobCategory.HEALTHCARE,
            'medical': JobCategory.HEALTHCARE,
            'nurse': JobCategory.HEALTHCARE,
            'manager': JobCategory.MANAGEMENT,
            'director': JobCategory.MANAGEMENT,
            'sales': JobCategory.SALES,
            'marketing': JobCategory.MARKETING,
            'design': JobCategory.DESIGN,
            'ux': JobCategory.DESIGN,
            'customer': JobCategory.CUSTOMER_SERVICE,
            'support': JobCategory.CUSTOMER_SERVICE,
            'teacher': JobCategory.EDUCATION,
            'educator': JobCategory.EDUCATION,
            'operations': JobCategory.OPERATIONS,
            'logistics': JobCategory.OPERATIONS
        }

    def analyze_job_market(self, query: str = None) -> Dict:
        """Analyze job market trends"""
        jobs = self.search_engine.jobs
        if not jobs:
            return {}

        analysis = {
            "market_overview": self._get_market_overview(jobs),
            "skill_demand": self._analyze_skill_demand(jobs),
            "salary_trends": self._analyze_salary_trends(jobs),
            "geographic_distribution": self._analyze_geographic_distribution(jobs),
            "remote_work_trends": self._analyze_remote_trends(jobs),
            "industry_insights": self._analyze_industries(jobs),
            "emerging_skills": self._find_emerging_skills(jobs),
            "job_clusters": self._analyze_job_clusters(),
            "recommendations": []
        }

        # Add query-specific insights
        if query:
            analysis["query_insights"] = self._get_query_insights(query, jobs)
            analysis["recommendations"] = self._generate_recommendations(query, analysis)

        return analysis

    def _get_market_overview(self, jobs: List[JobProfile]) -> Dict:
        """Get market overview"""
        total_jobs = len(jobs)

        # Count by source
        source_counts = Counter(job.source for job in jobs)

        # Count by job type
        job_type_counts = Counter(job.job_type for job in jobs if job.job_type)

        # Average posting recency
        recent_dates = []
        for job in jobs:
            try:
                date_obj = datetime.strptime(job.posted_date, '%Y-%m-%d')
                recent_dates.append((datetime.now() - date_obj).days)
            except:
                pass

        avg_recency = sum(recent_dates) / len(recent_dates) if recent_dates else 0

        return {
            "total_jobs": total_jobs,
            "sources": dict(source_counts),
            "job_types": dict(job_type_counts),
            "avg_posting_recency_days": round(avg_recency, 1)
        }

    def _analyze_skill_demand(self, jobs: List[JobProfile]) -> Dict:
        """Analyze skill demand across jobs"""
        all_skills = []
        for job in jobs:
            all_skills.extend(job.skills)

        skill_counts = Counter(all_skills)

        # Categorize skills
        skill_categories = {
            "programming": [],
            "frameworks": [],
            "cloud": [],
            "databases": [],
            "tools": [],
            "methodologies": [],
            "soft_skills": []
        }

        skill_patterns = {
            "programming": ["python", "java", "javascript", "typescript", "c++", "c#", "go", "rust", "php", "ruby"],
            "frameworks": ["react", "angular", "vue", "node", "django", "flask", "spring", "express", "laravel"],
            "cloud": ["aws", "azure", "gcp", "cloud", "devops", "docker", "kubernetes", "terraform"],
            "databases": ["sql", "nosql", "mongodb", "postgresql", "mysql", "redis", "oracle", "dynamodb"],
            "tools": ["git", "jenkins", "ansible", "grafana", "kafka", "airflow", "spark", "tableau"],
            "methodologies": ["agile", "scrum", "kanban", "ci/cd", "tdd", "microservices", "rest", "graphql"],
            "soft_skills": ["communication", "leadership", "teamwork", "problem solving", "analytical"]
        }

        for skill, count in skill_counts.most_common(100):
            skill_lower = skill.lower()
            categorized = False

            for category, patterns in skill_patterns.items():
                if any(pattern in skill_lower for pattern in patterns):
                    skill_categories[category].append((skill, count))
                    categorized = True
                    break

            if not categorized and len(skill.split()) == 1:  # Single word skills
                skill_categories["tools"].append((skill, count))

        # Limit each category
        for category in skill_categories:
            skill_categories[category] = skill_categories[category][:10]

        return {
            "top_skills": skill_counts.most_common(25),
            "by_category": skill_categories,
            "total_unique_skills": len(skill_counts)
        }

    def _analyze_salary_trends(self, jobs: List[JobProfile]) -> Dict:
        """Analyze salary trends"""
        salaries = []
        salaries_by_source = defaultdict(list)
        salaries_by_state = defaultdict(list)

        for job in jobs:
            if job.salary_avg:
                salaries.append(job.salary_avg)
                salaries_by_source[job.source].append(job.salary_avg)

                if job.state not in ['Remote', 'Unknown', 'USA']:
                    salaries_by_state[job.state].append(job.salary_avg)

        if not salaries:
            return {"message": "Insufficient salary data"}

        # Calculate statistics
        salary_stats = {
            "average": int(np.mean(salaries)),
            "median": int(np.median(salaries)),
            "min": int(np.min(salaries)),
            "max": int(np.max(salaries)),
            "std_dev": int(np.std(salaries)),
            "sample_size": len(salaries)
        }

        # By source
        source_stats = {}
        for source, source_salaries in salaries_by_source.items():
            if len(source_salaries) >= 3:
                source_stats[source] = {
                    "average": int(np.mean(source_salaries)),
                    "count": len(source_salaries)
                }

        # By state (top 10)
        state_stats = []
        for state, state_salaries in salaries_by_state.items():
            if len(state_salaries) >= 3:
                avg = np.mean(state_salaries)
                state_stats.append((state, int(avg), len(state_salaries)))

        state_stats.sort(key=lambda x: x[1], reverse=True)

        return {
            "overall": salary_stats,
            "by_source": source_stats,
            "top_paying_states": [(state, avg, count) for state, avg, count in state_stats[:10]]
        }

    def _analyze_geographic_distribution(self, jobs: List[JobProfile]) -> Dict:
        """Analyze geographic distribution"""
        state_counts = Counter(job.state for job in jobs)
        city_counts = Counter(job.city for job in jobs if job.city != 'Remote')

        # Remote jobs
        remote_count = sum(1 for job in jobs if job.remote_status == 'Remote')

        # By region
        regions = {
            'Northeast': ['CT', 'ME', 'MA', 'NH', 'RI', 'VT', 'NJ', 'NY', 'PA'],
            'Midwest': ['IL', 'IN', 'IA', 'KS', 'MI', 'MN', 'MO', 'NE', 'ND', 'OH', 'SD', 'WI'],
            'South': ['AL', 'AR', 'DE', 'FL', 'GA', 'KY', 'LA', 'MD', 'MS', 'NC', 'OK', 'SC', 'TN', 'TX', 'VA', 'WV'],
            'West': ['AK', 'AZ', 'CA', 'CO', 'HI', 'ID', 'MT', 'NV', 'NM', 'OR', 'UT', 'WA', 'WY']
        }

        region_counts = defaultdict(int)
        for state, count in state_counts.items():
            if state in ['Remote', 'Unknown', 'USA']:
                continue

            for region, states in regions.items():
                if state in states:
                    region_counts[region] += count
                    break

        return {
            "by_state": dict(state_counts),
            "by_city": dict(city_counts.most_common(20)),
            "by_region": dict(region_counts),
            "remote_jobs": remote_count,
            "remote_percentage": round((remote_count / len(jobs)) * 100, 1) if jobs else 0
        }

    def _analyze_remote_trends(self, jobs: List[JobProfile]) -> Dict:
        """Analyze remote work trends"""
        remote_by_source = defaultdict(int)
        total_by_source = defaultdict(int)

        for job in jobs:
            total_by_source[job.source] += 1
            if job.remote_status == 'Remote':
                remote_by_source[job.source] += 1

        remote_percentages = {}
        for source in total_by_source:
            if total_by_source[source] > 0:
                remote_percentages[source] = round(
                    (remote_by_source[source] / total_by_source[source]) * 100, 1
                )

        # Remote by job type
        remote_by_type = defaultdict(int)
        total_by_type = defaultdict(int)

        for job in jobs:
            if job.job_type:
                total_by_type[job.job_type] += 1
                if job.remote_status == 'Remote':
                    remote_by_type[job.job_type] += 1

        return {
            "remote_percentages": remote_percentages,
            "remote_by_type": {
                job_type: round((remote_by_type[job_type] / total) * 100, 1)
                for job_type, total in total_by_type.items() if total > 0
            },
            "total_remote": sum(remote_by_source.values()),
            "total_hybrid": sum(1 for job in jobs if job.remote_status == 'Hybrid')
        }

    def _analyze_industries(self, jobs: List[JobProfile]) -> Dict:
        """Analyze industry trends"""
        # Infer industry from company name and job title
        industry_keywords = {
            'Technology': ['tech', 'software', 'it', 'computer', 'data', 'cloud', 'cyber'],
            'Finance': ['bank', 'financial', 'insurance', 'capital', 'investment', 'wealth'],
            'Healthcare': ['health', 'medical', 'hospital', 'clinic', 'care', 'pharma'],
            'Retail': ['retail', 'store', 'shop', 'merchandise', 'e-commerce'],
            'Manufacturing': ['manufacturing', 'factory', 'production', 'industrial'],
            'Education': ['university', 'college', 'school', 'education', 'learning'],
            'Consulting': ['consulting', 'consultant', 'advisor', 'advisory']
        }

        industry_counts = defaultdict(int)

        for job in jobs:
            text = f"{job.company} {job.title}".lower()
            industry_found = False

            for industry, keywords in industry_keywords.items():
                if any(keyword in text for keyword in keywords):
                    industry_counts[industry] += 1
                    industry_found = True
                    break

            if not industry_found:
                industry_counts['Other'] += 1

        return dict(industry_counts)

    def _find_emerging_skills(self, jobs: List[JobProfile]) -> List[Tuple[str, int]]:
        """Find emerging skills (skills that appear in recent postings)"""
        # Get jobs from last 7 days
        recent_jobs = []
        for job in jobs:
            try:
                date_obj = datetime.strptime(job.posted_date, '%Y-%m-%d')
                if (datetime.now() - date_obj).days <= 7:
                    recent_jobs.append(job)
            except:
                pass

        if not recent_jobs:
            return []

        # Get skills from recent jobs
        recent_skills = []
        for job in recent_jobs:
            recent_skills.extend(job.skills)

        recent_skill_counts = Counter(recent_skills)

        # Get all skills for comparison
        all_skills = []
        for job in jobs:
            all_skills.extend(job.skills)

        all_skill_counts = Counter(all_skills)

        # Find skills that are more common in recent postings
        emerging = []
        for skill, recent_count in recent_skill_counts.most_common(20):
            total_count = all_skill_counts.get(skill, 0)
            if total_count > 10:  # Need enough data
                recent_ratio = recent_count / len(recent_jobs)
                total_ratio = total_count / len(jobs)

                if recent_ratio > total_ratio * 1.5:  # 50% more common recently
                    growth = ((recent_ratio - total_ratio) / total_ratio) * 100
                    emerging.append((skill, round(growth, 1), recent_count))

        emerging.sort(key=lambda x: x[1], reverse=True)
        return emerging[:10]

    def _analyze_job_clusters(self) -> Dict:
        """Analyze job clusters"""
        clusters = self.search_engine.cluster_jobs(8)

        cluster_analysis = {}
        for cluster_id, cluster_jobs in clusters.items():
            if len(cluster_jobs) < 3:
                continue

            # Find common features
            common_skills = Counter()
            common_titles = Counter()

            for job in cluster_jobs:
                common_skills.update(job.skills)
                common_titles.update([job.title])

            cluster_analysis[cluster_id] = {
                "size": len(cluster_jobs),
                "common_skills": common_skills.most_common(5),
                "common_titles": common_titles.most_common(3),
                "avg_salary": self._get_cluster_avg_salary(cluster_jobs),
                "remote_percentage": round(
                    sum(1 for j in cluster_jobs if j.remote_status == 'Remote') / len(cluster_jobs) * 100, 1
                )
            }

        return cluster_analysis

    def _get_cluster_avg_salary(self, jobs: List[JobProfile]) -> Optional[int]:
        """Get average salary for cluster"""
        salaries = [job.salary_avg for job in jobs if job.salary_avg]
        if salaries:
            return int(np.mean(salaries))
        return None

    def _get_query_insights(self, query: str, jobs: List[JobProfile]) -> Dict:
        """Get insights specific to query"""
        # Find matching jobs
        matching_jobs = self.search_engine.semantic_search(query, top_k=50)

        if not matching_jobs:
            return {"message": "No jobs match your query"}

        # Analyze matching jobs
        avg_salary = np.mean([j.salary_avg for j in matching_jobs if j.salary_avg]) if matching_jobs else 0
        remote_percentage = round(
            sum(1 for j in matching_jobs if j.remote_status == 'Remote') / len(matching_jobs) * 100, 1
        )

        # Common requirements
        common_skills = Counter()
        for job in matching_jobs:
            common_skills.update(job.skills)

        return {
            "matching_jobs": len(matching_jobs),
            "avg_salary": int(avg_salary) if avg_salary else None,
            "remote_percentage": remote_percentage,
            "top_skills_required": common_skills.most_common(10),
            "common_locations": Counter(job.full_location for job in matching_jobs).most_common(5),
            "top_companies": Counter(job.company for job in matching_jobs).most_common(5)
        }

    def _generate_recommendations(self, query: str, analysis: Dict) -> List[str]:
        """Generate personalized recommendations"""
        recommendations = []

        # Skill recommendations
        skill_demand = analysis.get("skill_demand", {})
        if "top_skills" in skill_demand:
            top_skills = [skill for skill, _ in skill_demand["top_skills"][:5]]
            recommendations.append(f"Focus on developing these high-demand skills: {', '.join(top_skills)}")

        # Location recommendations
        geo_dist = analysis.get("geographic_distribution", {})
        if "by_state" in geo_dist:
            top_states = sorted(geo_dist["by_state"].items(), key=lambda x: x[1], reverse=True)[:3]
            if top_states:
                states_str = ', '.join([state for state, _ in top_states])
                recommendations.append(f"Consider these locations with most opportunities: {states_str}")

        # Remote work recommendations
        remote_trends = analysis.get("remote_work_trends", {})
        if "remote_percentages" in remote_trends:
            best_source = max(remote_trends["remote_percentages"].items(), key=lambda x: x[1], default=None)
            if best_source:
                recommendations.append(f"For remote work, focus on {best_source[0]} ({best_source[1]}% remote jobs)")

        # Salary recommendations
        salary_trends = analysis.get("salary_trends", {})
        if "top_paying_states" in salary_trends:
            top_state = salary_trends["top_paying_states"][0] if salary_trends["top_paying_states"] else None
            if top_state:
                recommendations.append(f"Highest paying state: {top_state[0]} (avg ${top_state[1]:,})")

        # Emerging skills
        emerging = analysis.get("emerging_skills", [])
        if emerging:
            recommendations.append(f"Emerging skills to learn: {', '.join([skill for skill, _, _ in emerging[:3]])}")

        return recommendations

class RAGJobScraper:
    """Main RAG-powered job scraper orchestrator"""

    def __init__(self, use_llm: bool = False):
        self.search_engine = SemanticSearchEngine()
        self.analyzer = RAGJobAnalyzer(self.search_engine)
        self.use_llm = use_llm

        # Initialize website scrapers
        self.scrapers = {
            'Indeed': self._create_indeed_scraper(),
            'ZipRecruiter': self._create_ziprecruiter_scraper(),
            'LinkedIn': self._create_linkedin_scraper()
        }

        # LLM client if available
        if use_llm and LLM_AVAILABLE:
            self.llm_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        else:
            self.llm_client = None

        # Statistics
        self.stats = {
            'total_jobs': 0,
            'by_source': Counter(),
            'start_time': datetime.now(),
            'processed_urls': set()
        }

        # Output directory
        self.output_dir = Path("./rag_job_intelligence")
        self.output_dir.mkdir(exist_ok=True)

    def _create_indeed_scraper(self):
        """Create Indeed scraper with RAG enhancements"""
        class RAGIndeedScraper:
            def scrape(self, keyword, location, max_results=25):
                # Simplified Indeed scraping
                # In production, this would be a full scraper
                return []
        return RAGIndeedScraper()

    def _create_ziprecruiter_scraper(self):
        """Create ZipRecruiter scraper"""
        class RAGZipRecruiterScraper:
            def scrape(self, keyword, location, max_results=20):
                return []
        return RAGZipRecruiterScraper()

    def _create_linkedin_scraper(self):
        """Create LinkedIn scraper"""
        class RAGLinkedInScraper:
            def scrape(self, keyword, location, max_results=15):
                return []
        return RAGLinkedInScraper()

    def enrich_job_with_rag(self, job_data: Dict) -> JobProfile:
        """Enrich job data with RAG capabilities"""
        # Extract salary range
        salary_min, salary_max, salary_avg = self._parse_salary(job_data.get('salary', ''))

        # Generate embedding text
        description = job_data.get('description', '')[:1000]
        requirements = job_data.get('requirements', '')[:500]

        # Extract keywords
        keywords = self._extract_keywords(
            f"{job_data.get('title', '')} {description} {requirements}"
        )

        # Determine job category
        category = self._categorize_job(
            job_data.get('title', ''),
            job_data.get('skills', []),
            description
        )

        # Generate summary
        summary = self._generate_job_summary(
            job_data.get('title', ''),
            job_data.get('company', ''),
            job_data.get('skills', []),
            description[:200]
        )

        # Create job profile
        job = JobProfile(
            id=job_data.get('id', hashlib.md5(str(job_data).encode()).hexdigest()[:16]),
            title=job_data.get('title', ''),
            company=job_data.get('company', ''),
            city=job_data.get('city', ''),
            state=job_data.get('state', ''),
            full_location=job_data.get('full_location', ''),
            salary=job_data.get('salary', ''),
            salary_min=salary_min,
            salary_max=salary_max,
            salary_avg=salary_avg,
            experience=job_data.get('experience', ''),
            description=description,
            requirements=requirements,
            skills=job_data.get('skills', []),
            source=job_data.get('source', ''),
            url=job_data.get('url', ''),
            posted_date=job_data.get('posted_date', datetime.now().strftime('%Y-%m-%d')),
            job_type=job_data.get('job_type', ''),
            remote_status=job_data.get('remote_status', ''),
            industry=job_data.get('industry', ''),
            benefits=job_data.get('benefits', []),
            keywords=keywords,
            category=category,
            summary=summary,
            relevance_tags=self._generate_relevance_tags(
                job_data.get('title', ''),
                job_data.get('skills', []),
                description
            )
        )

        return job

    def _parse_salary(self, salary_text: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Parse salary text into min, max, avg"""
        if not salary_text or salary_text == "Not specified":
            return None, None, None

        # Extract numbers
        numbers = re.findall(r'\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?k?)', salary_text, re.IGNORECASE)

        if not numbers:
            return None, None, None

        # Convert to floats
        values = []
        for num in numbers:
            try:
                if num.lower().endswith('k'):
                    value = float(num[:-1].replace(',', '')) * 1000
                else:
                    value = float(num.replace(',', ''))
                values.append(value)
            except:
                continue

        if not values:
            return None, None, None

        if len(values) == 1:
            return values[0], values[0], values[0]
        else:
            return min(values), max(values), sum(values) / len(values)

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        # Remove common stopwords
        stopwords = {'the', 'and', 'for', 'with', 'this', 'that', 'are', 'was', 'were', 'have', 'has', 'had'}

        # Extract meaningful words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        keywords = [word for word in words if word not in stopwords]

        # Count frequencies
        keyword_counts = Counter(keywords)

        # Return top keywords
        return [keyword for keyword, _ in keyword_counts.most_common(10)]

    def _categorize_job(self, title: str, skills: List[str], description: str) -> JobCategory:
        """Categorize job based on title, skills, and description"""
        text = f"{title} {' '.join(skills)} {description}".lower()

        category_keywords = {
            JobCategory.TECH: ['software', 'developer', 'engineer', 'programmer', 'coder', 'tech', 'it'],
            JobCategory.DATA: ['data', 'analyst', 'scientist', 'analytics', 'bi', 'machine learning', 'ai'],
            JobCategory.FINANCE: ['finance', 'accountant', 'banking', 'investment', 'financial', 'audit'],
            JobCategory.HEALTHCARE: ['health', 'medical', 'nurse', 'doctor', 'hospital', 'clinic', 'care'],
            JobCategory.ENGINEERING: ['engineer', 'engineering', 'mechanical', 'electrical', 'civil'],
            JobCategory.MARKETING: ['marketing', 'brand', 'advertising', 'social media', 'digital marketing'],
            JobCategory.SALES: ['sales', 'account executive', 'business development', 'sales representative'],
            JobCategory.DESIGN: ['design', 'designer', 'ux', 'ui', 'graphic', 'creative'],
            JobCategory.MANAGEMENT: ['manager', 'director', 'lead', 'head of', 'vp', 'chief'],
            JobCategory.OPERATIONS: ['operations', 'logistics', 'supply chain', 'production', 'manufacturing'],
            JobCategory.CUSTOMER_SERVICE: ['customer', 'support', 'service', 'help desk', 'client'],
            JobCategory.EDUCATION: ['teacher', 'professor', 'educator', 'instructor', 'tutor', 'education']
        }

        for category, keywords in category_keywords.items():
            if any(keyword in text for keyword in keywords):
                return category

        return JobCategory.OTHER

    def _generate_job_summary(self, title: str, company: str, skills: List[str], description: str) -> str:
        """Generate a concise job summary"""
        if self.llm_client:
            try:
                prompt = f"""
                Generate a concise summary (max 2 sentences) for this job:
                Title: {title}
                Company: {company}
                Key Skills: {', '.join(skills[:5])}
                Description: {description[:300]}

                Summary:
                """

                response = self.llm_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100,
                    temperature=0.3
                )

                return response.choices[0].message.content.strip()
            except:
                pass

        # Fallback summary
        return f"{title} position at {company} requiring skills in {', '.join(skills[:3])}."

    def _generate_relevance_tags(self, title: str, skills: List[str], description: str) -> List[str]:
        """Generate relevance tags for job"""
        tags = []

        # Experience level tags
        exp_text = f"{title} {description}".lower()
        if 'senior' in exp_text or 'lead' in exp_text or 'principal' in exp_text:
            tags.append('senior')
        elif 'junior' in exp_text or 'entry' in exp_text:
            tags.append('entry-level')
        elif 'mid' in exp_text or 'intermediate' in exp_text:
            tags.append('mid-level')

        # Technology tags
        tech_keywords = {
            'frontend': ['react', 'angular', 'vue', 'javascript', 'typescript', 'html', 'css'],
            'backend': ['python', 'java', 'node', 'django', 'flask', 'spring', 'api'],
            'devops': ['aws', 'azure', 'docker', 'kubernetes', 'terraform', 'ci/cd'],
            'data': ['sql', 'mongodb', 'postgresql', 'data analysis', 'machine learning'],
            'mobile': ['ios', 'android', 'swift', 'kotlin', 'react native']
        }

        for tag, keywords in tech_keywords.items():
            if any(keyword in exp_text for keyword in keywords):
                tags.append(tag)

        # Remote tag
        if 'remote' in exp_text or 'work from home' in exp_text:
            tags.append('remote')

        return list(set(tags))[:5]

    def search_jobs(self, query: str, filters: Dict = None, top_k: int = 20) -> Dict:
        """Search jobs using RAG"""
        print(f"\n🔍 Performing RAG search for: '{query}'")

        # Semantic search
        results = self.search_engine.hybrid_search(query, filters, top_k)

        # Generate insights
        insights = self._generate_search_insights(query, results)

        return {
            "query": query,
            "results": results,
            "count": len(results),
            "insights": insights,
            "filters_applied": filters or {}
        }

    def _generate_search_insights(self, query: str, results: List[JobProfile]) -> Dict:
        """Generate insights for search results"""
        if not results:
            return {"message": "No results found"}

        insights = {
            "top_skills_in_results": Counter(
                skill for job in results for skill in job.skills
            ).most_common(10),
            "common_companies": Counter(job.company for job in results).most_common(5),
            "location_distribution": Counter(job.full_location for job in results).most_common(5),
            "salary_range": self._get_results_salary_range(results),
            "remote_percentage": round(
                sum(1 for job in results if job.remote_status == 'Remote') / len(results) * 100, 1
            )
        }

        # Add semantic match distribution
        score_bins = {'High (>0.7)': 0, 'Medium (0.4-0.7)': 0, 'Low (<0.4)': 0}
        for job in results:
            if job.semantic_score > 0.7:
                score_bins['High (>0.7)'] += 1
            elif job.semantic_score > 0.4:
                score_bins['Medium (0.4-0.7)'] += 1
            else:
                score_bins['Low (<0.4)'] += 1

        insights["match_quality"] = score_bins

        return insights

    def _get_results_salary_range(self, results: List[JobProfile]) -> Dict:
        """Get salary range from results"""
        salaries = [job.salary_avg for job in results if job.salary_avg]

        if not salaries:
            return {"message": "No salary data available"}

        return {
            "average": int(np.mean(salaries)),
            "min": int(np.min(salaries)),
            "max": int(np.max(salaries)),
            "sample_size": len(salaries)
        }

    def get_personalized_recommendations(self, user_profile: Dict) -> Dict:
        """Get personalized job recommendations"""
        print(f"\n🎯 Generating personalized recommendations...")

        # Build user query from profile
        query_parts = []

        if 'skills' in user_profile:
            query_parts.append(f"Skills: {', '.join(user_profile['skills'][:5])}")

        if 'experience' in user_profile:
            query_parts.append(f"Experience: {user_profile['experience']}")

        if 'preferred_roles' in user_profile:
            query_parts.append(f"Roles: {', '.join(user_profile['preferred_roles'][:3])}")

        if 'preferred_location' in user_profile:
            query_parts.append(f"Location: {user_profile['preferred_location']}")

        query = " ".join(query_parts)

        # Build filters
        filters = {}
        if 'preferred_locations' in user_profile:
            filters['locations'] = user_profile['preferred_locations']

        if 'remote_only' in user_profile and user_profile['remote_only']:
            filters['remote_only'] = True

        if 'min_salary' in user_profile:
            filters['min_salary'] = user_profile['min_salary']

        if 'experience_level' in user_profile:
            filters['experience_level'] = user_profile['experience_level']

        # Search for matching jobs
        search_results = self.search_jobs(query, filters, top_k=30)

        # Analyze gaps and opportunities
        analysis = self.analyzer.analyze_job_market(query)

        # Generate personalized advice
        advice = self._generate_personalized_advice(user_profile, search_results, analysis)

        return {
            "user_profile": {k: v for k, v in user_profile.items() if k != 'skills'},  # Anonymize
            "recommended_jobs": search_results["results"][:15],
            "market_analysis": analysis,
            "personalized_advice": advice,
            "skill_gaps": self._identify_skill_gaps(user_profile.get('skills', []), analysis),
            "next_steps": self._generate_next_steps(user_profile, analysis)
        }

    def _generate_personalized_advice(self, user_profile: Dict, search_results: Dict, analysis: Dict) -> List[str]:
        """Generate personalized advice"""
        advice = []

        # Check if user has required skills
        user_skills = set(skill.lower() for skill in user_profile.get('skills', []))

        if search_results["results"]:
            # Get top skills from matching jobs
            top_job_skills = Counter()
            for job in search_results["results"][:10]:
                top_job_skills.update(job.skills)

            missing_skills = []
            for skill, count in top_job_skills.most_common(10):
                if skill.lower() not in user_skills:
                    missing_skills.append(skill)

            if missing_skills:
                advice.append(f"Consider learning: {', '.join(missing_skills[:3])}")

        # Location advice
        if 'preferred_location' in user_profile:
            loc = user_profile['preferred_location']
            geo_data = analysis.get("geographic_distribution", {})

            if 'by_state' in geo_data:
                state_count = geo_data['by_state'].get(loc, 0)
                if state_count < 10:
                    advice.append(f"Few opportunities in {loc}. Consider expanding search to nearby states.")

        # Salary advice
        user_exp = user_profile.get('experience', '')
        salary_data = analysis.get("salary_trends", {})

        if salary_data.get("overall", {}).get("average"):
            avg_salary = salary_data["overall"]["average"]

            if 'entry' in user_exp.lower() or 'junior' in user_exp.lower():
                advice.append(f"Entry-level average salary: ${avg_salary:,}")
            elif 'senior' in user_exp.lower():
                advice.append(f"Senior-level positions average: ${avg_salary:,}")

        return advice[:5]

    def _identify_skill_gaps(self, user_skills: List[str], analysis: Dict) -> List[Tuple[str, int]]:
        """Identify skill gaps between user skills and market demand"""
        skill_demand = analysis.get("skill_demand", {}).get("top_skills", [])

        user_skill_set = set(skill.lower() for skill in user_skills)

        gaps = []
        for skill, demand_count in skill_demand[:20]:
            if skill.lower() not in user_skill_set:
                gaps.append((skill, demand_count))

        return gaps[:10]

    def _generate_next_steps(self, user_profile: Dict, analysis: Dict) -> List[str]:
        """Generate next steps for job search"""
        steps = []

        # Skill development
        emerging_skills = analysis.get("emerging_skills", [])
        if emerging_skills:
            steps.append(f"Learn emerging skill: {emerging_skills[0][0]}")

        # Location strategy
        geo_data = analysis.get("geographic_distribution", {})
        if 'by_state' in geo_data:
            top_state = max(geo_data['by_state'].items(), key=lambda x: x[1], default=None)
            if top_state and top_state[0] not in ['Remote', 'Unknown']:
                steps.append(f"Search in {top_state[0]} (most opportunities)")

        # Remote work
        remote_trends = analysis.get("remote_work_trends", {})
        if remote_trends.get("remote_percentages"):
            best_remote_source = max(
                remote_trends["remote_percentages"].items(),
                key=lambda x: x[1],
                default=None
            )
            if best_remote_source:
                steps.append(f"Use {best_remote_source[0]} for remote opportunities")

        return steps

    def save_rag_results(self, search_results: Dict, analysis: Dict, filename_prefix: str = "rag_search"):
        """Save RAG search results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save search results
        results_data = {
            "metadata": {
                "query": search_results.get("query", ""),
                "count": search_results.get("count", 0),
                "timestamp": timestamp,
                "filters": search_results.get("filters_applied", {})
            },
            "insights": search_results.get("insights", {}),
            "jobs": [job.to_dict() for job in search_results.get("results", [])]
        }

        results_file = self.output_dir / f"{filename_prefix}_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)

        # Save analysis
        analysis_file = self.output_dir / f"{filename_prefix}_analysis_{timestamp}.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)

        # Save embeddings if available
        if self.search_engine.embeddings is not None:
            embeddings_file = self.output_dir / f"{filename_prefix}_embeddings_{timestamp}.pkl"
            with open(embeddings_file, 'wb') as f:
                pickle.dump({
                    "embeddings": self.search_engine.embeddings,
                    "job_ids": [job.id for job in self.search_engine.jobs]
                }, f)

        print(f"\n💾 RAG results saved:")
        print(f"   📊 Search results: {results_file}")
        print(f"   📈 Market analysis: {analysis_file}")

        if self.search_engine.embeddings is not None:
            print(f"   🔧 Embeddings: {embeddings_file}")

class RAGJobIntelligenceSystem:
    """Complete RAG-powered job intelligence system"""

    def __init__(self):
        self.rag_scraper = RAGJobScraper(use_llm=False)
        self.demo_data_loaded = False

    def load_demo_data(self):
        """Load demo data for testing RAG capabilities"""
        print("\n📂 Loading demo job data for RAG testing...")

        demo_jobs = self._generate_demo_jobs()

        for job_data in demo_jobs:
            job_profile = self.rag_scraper.enrich_job_with_rag(job_data)
            self.rag_scraper.search_engine.add_job(job_profile)

        # Build indexes
        self.rag_scraper.search_engine.build_index()

        self.demo_data_loaded = True
        print(f"✅ Loaded {len(demo_jobs)} demo jobs with RAG enrichment")

        # Show sample enriched job
        if demo_jobs:
            sample = self.rag_scraper.search_engine.jobs[0]
            print(f"\n📋 Sample enriched job:")
            print(f"   Title: {sample.title}")
            print(f"   Category: {sample.category.value}")
            print(f"   Skills: {', '.join(sample.skills[:5])}")
            print(f"   Summary: {sample.summary}")

    def _generate_demo_jobs(self) -> List[Dict]:
        """Generate demo job data"""
        return [
            {
                "id": "job_001",
                "title": "Senior Python Developer",
                "company": "TechCorp Inc",
                "city": "San Francisco",
                "state": "CA",
                "full_location": "San Francisco, CA",
                "salary": "$120,000 - $160,000 per year",
                "experience": "5+ years Python experience",
                "description": "We're looking for a Senior Python Developer with experience in Django, Flask, and AWS. You'll be building scalable web applications and working with our data science team.",
                "requirements": "5+ years Python, Django/Flask, AWS, SQL, REST APIs",
                "skills": ["Python", "Django", "Flask", "AWS", "SQL", "REST API"],
                "source": "Indeed",
                "url": "https://indeed.com/job1",
                "posted_date": "2024-01-15",
                "job_type": "Full-time",
                "remote_status": "Hybrid",
                "industry": "Technology"
            },
            {
                "id": "job_002",
                "title": "Data Scientist",
                "company": "DataAnalytics Co",
                "city": "New York",
                "state": "NY",
                "full_location": "New York, NY",
                "salary": "$130,000 - $180,000",
                "experience": "3+ years in data science",
                "description": "Join our data science team to build machine learning models and analyze large datasets. Experience with TensorFlow and PyTorch required.",
                "requirements": "Python, Machine Learning, TensorFlow, SQL, Statistics",
                "skills": ["Python", "Machine Learning", "TensorFlow", "PyTorch", "SQL", "Statistics"],
                "source": "LinkedIn",
                "url": "https://linkedin.com/job2",
                "posted_date": "2024-01-14",
                "job_type": "Full-time",
                "remote_status": "Remote",
                "industry": "Data Analytics"
            },
            {
                "id": "job_003",
                "title": "DevOps Engineer",
                "company": "CloudSystems LLC",
                "city": "Austin",
                "state": "TX",
                "full_location": "Austin, TX",
                "salary": "$110,000 - $150,000",
                "experience": "4+ years DevOps experience",
                "description": "We need a DevOps Engineer to manage our AWS infrastructure, implement CI/CD pipelines, and ensure system reliability.",
                "requirements": "AWS, Docker, Kubernetes, Terraform, CI/CD, Linux",
                "skills": ["AWS", "Docker", "Kubernetes", "Terraform", "CI/CD", "Linux"],
                "source": "ZipRecruiter",
                "url": "https://ziprecruiter.com/job3",
                "posted_date": "2024-01-13",
                "job_type": "Full-time",
                "remote_status": "On-site",
                "industry": "Cloud Computing"
            },
            {
                "id": "job_004",
                "title": "Frontend React Developer",
                "company": "WebSolutions Inc",
                "city": "Remote",
                "state": "USA",
                "full_location": "Remote, USA",
                "salary": "$90,000 - $130,000",
                "experience": "3+ years React development",
                "description": "Build beautiful user interfaces with React, TypeScript, and modern frontend tools. Fully remote position.",
                "requirements": "React, TypeScript, JavaScript, HTML, CSS, REST APIs",
                "skills": ["React", "TypeScript", "JavaScript", "HTML", "CSS", "REST API"],
                "source": "Indeed",
                "url": "https://indeed.com/job4",
                "posted_date": "2024-01-12",
                "job_type": "Full-time",
                "remote_status": "Remote",
                "industry": "Web Development"
            },
            {
                "id": "job_005",
                "title": "Machine Learning Engineer",
                "company": "AI Innovations",
                "city": "Seattle",
                "state": "WA",
                "full_location": "Seattle, WA",
                "salary": "$140,000 - $200,000",
                "experience": "4+ years ML engineering",
                "description": "Work on cutting-edge AI projects. Experience with LLMs, vector databases, and MLOps required.",
                "requirements": "Python, Machine Learning, LLM, MLOps, AWS, Docker",
                "skills": ["Python", "Machine Learning", "LLM", "MLOps", "AWS", "Docker"],
                "source": "LinkedIn",
                "url": "https://linkedin.com/job5",
                "posted_date": "2024-01-11",
                "job_type": "Full-time",
                "remote_status": "Hybrid",
                "industry": "Artificial Intelligence"
            }
        ]

    def run_rag_demo(self):
        """Run RAG capabilities demo"""
        print("\n" + "=" * 70)
        print("🧠 RAG-POWERED JOB INTELLIGENCE DEMO")
        print("=" * 70)

        if not self.demo_data_loaded:
            self.load_demo_data()

        # Demo 1: Semantic Search
        print("\n1. 🔍 SEMANTIC JOB SEARCH")
        print("-" * 40)

        queries = [
            "Python developer with cloud experience",
            "Remote data science positions",
            "Senior engineer AWS Docker",
            "Machine learning AI jobs"
        ]

        for query in queries:
            print(f"\n   Searching: '{query}'")
            results = self.rag_scraper.search_jobs(query, top_k=3)

            if results["results"]:
                for i, job in enumerate(results["results"][:2], 1):
                    print(f"   {i}. {job.title} at {job.company}")
                    print(f"      Match: {job.semantic_score:.2f}")
                    print(f"      Reasons: {', '.join(job.match_reasons[:2])}")
            else:
                print("   No results found")

        # Demo 2: Market Analysis
        print("\n\n2. 📊 MARKET INTELLIGENCE")
        print("-" * 40)

        analysis = self.rag_scraper.analyzer.analyze_job_market("technology jobs")

        if analysis:
            print(f"   Total jobs analyzed: {analysis.get('market_overview', {}).get('total_jobs', 0)}")

            skill_demand = analysis.get('skill_demand', {}).get('top_skills', [])
            if skill_demand:
                print(f"   Top skills in demand:")
                for skill, count in skill_demand[:5]:
                    print(f"      • {skill}: {count}")

            salary_trends = analysis.get('salary_trends', {}).get('overall', {})
            if salary_trends.get('average'):
                print(f"   Average salary: ${salary_trends['average']:,}")

        # Demo 3: Personalized Recommendations
        print("\n\n3. 🎯 PERSONALIZED RECOMMENDATIONS")
        print("-" * 40)

        user_profile = {
            "skills": ["Python", "SQL", "AWS"],
            "experience": "3 years",
            "preferred_roles": ["Data Scientist", "ML Engineer"],
            "preferred_location": "Remote",
            "remote_only": True,
            "min_salary": 100000
        }

        recommendations = self.rag_scraper.get_personalized_recommendations(user_profile)

        print(f"   For user with skills: {', '.join(user_profile['skills'])}")
        print(f"   Found {len(recommendations.get('recommended_jobs', []))} matching jobs")

        if recommendations.get('personalized_advice'):
            print(f"   Personalized advice:")
            for advice in recommendations['personalized_advice'][:3]:
                print(f"      • {advice}")

        # Demo 4: Similar Jobs
        print("\n\n4. 🔄 FIND SIMILAR JOBS")
        print("-" * 40)

        if self.rag_scraper.search_engine.jobs:
            sample_job = self.rag_scraper.search_engine.jobs[0]
            print(f"   Finding jobs similar to: {sample_job.title}")

            similar_jobs = self.rag_scraper.search_engine.find_similar_jobs(sample_job.id, top_k=3)

            for i, job in enumerate(similar_jobs, 1):
                print(f"   {i}. {job.title} at {job.company}")
                print(f"      Similarity: {job.semantic_score:.2f}")

        # Demo 5: Job Clusters
        print("\n\n5. 📁 JOB CLUSTERS")
        print("-" * 40)

        clusters = self.rag_scraper.search_engine.cluster_jobs(3)

        for cluster_id, cluster_jobs in clusters.items():
            if cluster_jobs:
                print(f"   Cluster {cluster_id}: {len(cluster_jobs)} jobs")
                print(f"      Sample titles: {', '.join([j.title[:20] + '...' for j in cluster_jobs[:2]])}")

    def interactive_rag_search(self):
        """Interactive RAG search interface"""
        print("\n" + "=" * 70)
        print("💬 INTERACTIVE RAG JOB SEARCH")
        print("=" * 70)

        if not self.demo_data_loaded:
            self.load_demo_data()

        while True:
            print("\nOptions:")
            print("1. 🔍 Search jobs")
            print("2. 📊 Analyze job market")
            print("3. 🎯 Get personalized recommendations")
            print("4. 💾 Save current data")
            print("5. 🚪 Exit")

            choice = input("\nEnter choice (1-5): ").strip()

            if choice == '1':
                query = input("Enter job search query: ").strip()
                if query:
                    results = self.rag_scraper.search_jobs(query, top_k=10)

                    if results["results"]:
                        print(f"\nFound {results['count']} jobs:")
                        for i, job in enumerate(results["results"][:5], 1):
                            print(f"\n{i}. {job.title} at {job.company}")
                            print(f"   Location: {job.full_location}")
                            print(f"   Salary: {job.salary}")
                            print(f"   Match Score: {job.semantic_score:.2f}")
                            print(f"   Match Reasons: {', '.join(job.match_reasons[:2])}")

                            save = input(f"\nSave these results? (y/n): ").lower()
                            if save == 'y':
                                self.rag_scraper.save_rag_results(results, {}, f"search_{query[:20]}")
                    else:
                        print("No jobs found.")

            elif choice == '2':
                query = input("Enter analysis topic (optional): ").strip()
                analysis = self.rag_scraper.analyzer.analyze_job_market(query if query else None)

                if analysis:
                    print("\n📈 MARKET ANALYSIS:")
                    print(f"Total jobs: {analysis.get('market_overview', {}).get('total_jobs', 0)}")

                    # Show top skills
                    skills = analysis.get('skill_demand', {}).get('top_skills', [])
                    if skills:
                        print("\nTop 5 Skills in Demand:")
                        for skill, count in skills[:5]:
                            print(f"  • {skill}: {count} mentions")

                    # Show salary info
                    salary = analysis.get('salary_trends', {}).get('overall', {})
                    if salary.get('average'):
                        print(f"\nAverage Salary: ${salary['average']:,}")
                        print(f"Range: ${salary['min']:,} - ${salary['max']:,}")

            elif choice == '3':
                print("\nCreate your profile:")
                skills = input("Your skills (comma-separated): ").strip().split(',')
                experience = input("Your experience level: ").strip()
                location = input("Preferred location: ").strip()

                user_profile = {
                    "skills": [s.strip() for s in skills if s.strip()],
                    "experience": experience,
                    "preferred_location": location,
                    "remote_only": False,
                    "min_salary": 0
                }

                recs = self.rag_scraper.get_personalized_recommendations(user_profile)

                if recs.get("recommended_jobs"):
                    print(f"\n🎯 RECOMMENDED JOBS:")
                    for i, job in enumerate(recs["recommended_jobs"][:3], 1):
                        print(f"\n{i}. {job.title} at {job.company}")
                        print(f"   Match Score: {job.semantic_score:.2f}")

                if recs.get("personalized_advice"):
                    print("\n💡 PERSONALIZED ADVICE:")
                    for advice in recs["personalized_advice"]:
                        print(f"  • {advice}")

            elif choice == '4':
                # Save all current data
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # Save jobs
                jobs_data = [job.to_dict() for job in self.rag_scraper.search_engine.jobs]
                jobs_file = self.rag_scraper.output_dir / f"rag_jobs_{timestamp}.json"
                with open(jobs_file, 'w') as f:
                    json.dump(jobs_data, f, indent=2)

                print(f"💾 Saved {len(jobs_data)} jobs to {jobs_file}")

            elif choice == '5':
                print("Goodbye!")
                break

            else:
                print("Invalid choice. Please try again.")

def main():
    """Main function to run the RAG job intelligence system"""
    print("\n" + "=" * 70)
    print("🧠 RAG-POWERED JOB INTELLIGENCE SYSTEM")
    print("=" * 70)

    print("\nFeatures:")
    print("  • Semantic job search with embeddings")
    print("  • Intelligent job matching")
    print("  • Market trend analysis")
    print("  • Personalized recommendations")
    print("  • Skill gap analysis")
    print("  • Job clustering and categorization")

    print("\nRequired packages:")
    print("  pip install sentence-transformers scikit-learn pandas numpy")
    print("\nOptional for LLM features:")
    print("  pip install openai")

    # Initialize system
    system = RAGJobIntelligenceSystem()

    print("\nStarting in 3 seconds...")
    time.sleep(3)

    # Run demo
    system.run_rag_demo()

    # Interactive mode
    system.interactive_rag_search()

if __name__ == "__main__":
    main()
