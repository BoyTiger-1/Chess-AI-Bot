"""
User and organization models.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Text, Table
from sqlalchemy.orm import relationship

from ai_business_assistant.shared.database import Base


class User(Base):
    """User model with authentication and profile information."""
    
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    
    first_name = Column(String(100))
    last_name = Column(String(100))
    avatar_url = Column(String(500))
    
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    is_superuser = Column(Boolean, default=False)
    
    mfa_enabled = Column(Boolean, default=False)
    mfa_secret = Column(String(100))
    
    oauth_provider = Column(String(50))
    oauth_id = Column(String(255))
    
    last_login = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    organizations = relationship(
        "Organization",
        secondary="user_organizations",
        back_populates="users"
    )
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User {self.email}>"


class Organization(Base):
    """Organization/tenant model for multi-tenancy."""
    
    __tablename__ = "organizations"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    slug = Column(String(100), unique=True, index=True, nullable=False)
    
    description = Column(Text)
    logo_url = Column(String(500))
    website = Column(String(255))
    
    is_active = Column(Boolean, default=True)
    subscription_tier = Column(String(50), default="free")
    subscription_expires_at = Column(DateTime)
    
    settings = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    users = relationship(
        "User",
        secondary="user_organizations",
        back_populates="organizations"
    )
    
    def __repr__(self):
        return f"<Organization {self.name}>"


class UserOrganization(Base):
    """Association table for user-organization many-to-many relationship."""
    
    __tablename__ = "user_organizations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False)
    
    role = Column(String(50), default="member")
    is_default = Column(Boolean, default=False)
    
    joined_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<UserOrganization user_id={self.user_id} org_id={self.organization_id}>"
