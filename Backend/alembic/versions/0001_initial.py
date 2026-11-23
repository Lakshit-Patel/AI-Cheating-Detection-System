"""initial production schema

Revision ID: 0001_initial
Revises:
"""
from alembic import op
import sqlalchemy as sa

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("role", sa.String(length=20), nullable=False, server_default="student"),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("is_active", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("role IN ('student', 'admin')", name="ck_users_role"),
        sa.CheckConstraint("is_active IN (0, 1)", name="ck_users_active"),
        sa.UniqueConstraint("email", name="uq_users_email"),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=False)

    op.create_table(
        "exam_sessions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("student_id", sa.Integer(), nullable=False),
        sa.Column("start_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("end_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="ACTIVE"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("status IN ('ACTIVE', 'COMPLETED', 'CANCELLED')", name="ck_exam_sessions_status"),
        sa.ForeignKeyConstraint(["student_id"], ["users.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_exam_sessions_student_status", "exam_sessions", ["student_id", "status"])
    op.create_index("ix_exam_sessions_start_time", "exam_sessions", ["start_time"])

    op.create_table(
        "violations",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("session_id", sa.Integer(), nullable=False),
        sa.Column("violation_type", sa.String(length=100), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column("details", sa.Text(), nullable=True),
        sa.CheckConstraint("confidence IS NULL OR (confidence >= 0 AND confidence <= 1)", name="ck_violations_confidence"),
        sa.ForeignKeyConstraint(["session_id"], ["exam_sessions.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_violations_session_timestamp", "violations", ["session_id", "timestamp"])
    op.create_index("ix_violations_type_timestamp", "violations", ["violation_type", "timestamp"])


def downgrade():
    op.drop_index("ix_violations_type_timestamp", table_name="violations")
    op.drop_index("ix_violations_session_timestamp", table_name="violations")
    op.drop_table("violations")
    op.drop_index("ix_exam_sessions_start_time", table_name="exam_sessions")
    op.drop_index("ix_exam_sessions_student_status", table_name="exam_sessions")
    op.drop_table("exam_sessions")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
