"""
Role-based access control models.
"""

from datetime import datetime

from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship

from ai_business_assistant.shared.database import Base


class Role(Base):
    """Role model for RBAC."""
    
    __tablename__ = "roles"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text)
    is_system = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    permissions = relationship(
        "Permission",
        secondary="role_permissions",
        back_populates="roles"
    )


class Permission(Base):
    """Permission model for RBAC."""
    
    __tablename__ = "permissions"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    resource = Column(String(100), nullable=False)
    action = Column(String(50), nullable=False)
    description = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    roles = relationship(
        "Role",
        secondary="role_permissions",
        back_populates="permissions"
    )


class RolePermission(Base):
    """Association table for role-permission many-to-many relationship."""
    
    __tablename__ = "role_permissions"
    
    id = Column(Integer, primary_key=True, index=True)
    role_id = Column(Integer, ForeignKey("roles.id"), nullable=False)
    permission_id = Column(Integer, ForeignKey("permissions.id"), nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
