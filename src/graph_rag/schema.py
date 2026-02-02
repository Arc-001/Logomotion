"""
Knowledge Graph Schema for Manim Code.

Defines the structure of nodes and relationships in the knowledge graph:
- ManimClass: Classes like Scene, Circle, Square
- ManimFunction: Functions and methods  
- Animation: Animation types like Write, FadeIn, Transform
- Import: Import statements and modules
- Example: Complete code examples from the dataset
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class NodeType(str, Enum):
    """Types of nodes in the Manim knowledge graph."""
    MANIM_CLASS = "ManimClass"
    MANIM_FUNCTION = "ManimFunction"
    ANIMATION = "Animation"
    IMPORT = "Import"
    EXAMPLE = "Example"
    CONCEPT = "Concept"  # Mathematical/visual concepts like "vector field", "3D"


class RelationType(str, Enum):
    """Types of relationships between nodes."""
    USES = "USES"  # Example uses a class/function
    INHERITS_FROM = "INHERITS_FROM"  # Class inheritance
    ANIMATES = "ANIMATES"  # Animation applied to object
    IMPORTS = "IMPORTS"  # Code imports module
    SIMILAR_TO = "SIMILAR_TO"  # Semantic similarity
    DEMONSTRATES = "DEMONSTRATES"  # Example demonstrates concept
    BELONGS_TO = "BELONGS_TO"  # Function belongs to class


class ManimClassNode(BaseModel):
    """Represents a Manim class (Scene, Circle, etc.)."""
    name: str = Field(..., description="Class name")
    module: str = Field(default="manim", description="Module containing the class")
    description: Optional[str] = Field(None, description="What this class does")
    is_scene: bool = Field(default=False, description="Whether this is a Scene subclass")
    is_mobject: bool = Field(default=False, description="Whether this is a Mobject")


class ManimFunctionNode(BaseModel):
    """Represents a Manim function or method."""
    name: str = Field(..., description="Function name")
    parent_class: Optional[str] = Field(None, description="Parent class if method")
    signature: Optional[str] = Field(None, description="Function signature")
    description: Optional[str] = Field(None, description="What this function does")


class AnimationNode(BaseModel):
    """Represents a Manim animation type."""
    name: str = Field(..., description="Animation name (Write, FadeIn, etc.)")
    description: Optional[str] = Field(None, description="What this animation does")
    duration_default: Optional[float] = Field(None, description="Default duration")


class ImportNode(BaseModel):
    """Represents an import statement."""
    statement: str = Field(..., description="Full import statement")
    module: str = Field(..., description="Module being imported")


class ExampleNode(BaseModel):
    """Represents a complete code example from the dataset."""
    id: str = Field(..., description="Unique identifier")
    prompt: str = Field(..., description="User prompt that generated this code")
    code: str = Field(..., description="The generated Manim code")
    scene_class: Optional[str] = Field(None, description="Main scene class name")
    embedding_id: Optional[str] = Field(None, description="ID in vector store")


class ConceptNode(BaseModel):
    """Represents a mathematical or visual concept."""
    name: str = Field(..., description="Concept name")
    category: str = Field(default="general", description="Category (math, physics, etc.)")
    description: Optional[str] = Field(None, description="Concept description")


# Cypher queries for Neo4j schema creation
SCHEMA_CONSTRAINTS = """
CREATE CONSTRAINT manim_class_name IF NOT EXISTS FOR (n:ManimClass) REQUIRE n.name IS UNIQUE;
CREATE CONSTRAINT animation_name IF NOT EXISTS FOR (n:Animation) REQUIRE n.name IS UNIQUE;
CREATE CONSTRAINT example_id IF NOT EXISTS FOR (n:Example) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (n:Concept) REQUIRE n.name IS UNIQUE;
"""

SCHEMA_INDEXES = """
CREATE INDEX example_prompt IF NOT EXISTS FOR (n:Example) ON (n.prompt);
CREATE INDEX manim_class_module IF NOT EXISTS FOR (n:ManimClass) ON (n.module);
CREATE FULLTEXT INDEX example_code IF NOT EXISTS FOR (n:Example) ON EACH [n.code, n.prompt];
"""


# Common Manim classes for initial knowledge graph population
KNOWN_MANIM_CLASSES = [
    ManimClassNode(name="Scene", module="manim", is_scene=True, description="Base class for all scenes"),
    ManimClassNode(name="ThreeDScene", module="manim", is_scene=True, description="Scene for 3D animations"),
    ManimClassNode(name="Circle", module="manim", is_mobject=True, description="A circle shape"),
    ManimClassNode(name="Square", module="manim", is_mobject=True, description="A square shape"),
    ManimClassNode(name="Rectangle", module="manim", is_mobject=True, description="A rectangle shape"),
    ManimClassNode(name="Line", module="manim", is_mobject=True, description="A straight line"),
    ManimClassNode(name="Arrow", module="manim", is_mobject=True, description="An arrow"),
    ManimClassNode(name="Vector", module="manim", is_mobject=True, description="A vector arrow"),
    ManimClassNode(name="Dot", module="manim", is_mobject=True, description="A point/dot"),
    ManimClassNode(name="Text", module="manim", is_mobject=True, description="Text object"),
    ManimClassNode(name="Tex", module="manim", is_mobject=True, description="LaTeX text"),
    ManimClassNode(name="MathTex", module="manim", is_mobject=True, description="LaTeX math expression"),
    ManimClassNode(name="VGroup", module="manim", is_mobject=True, description="Group of vector mobjects"),
    ManimClassNode(name="Axes", module="manim", is_mobject=True, description="2D coordinate axes"),
    ManimClassNode(name="ThreeDAxes", module="manim", is_mobject=True, description="3D coordinate axes"),
    ManimClassNode(name="NumberPlane", module="manim", is_mobject=True, description="Coordinate plane grid"),
    ManimClassNode(name="ComplexPlane", module="manim", is_mobject=True, description="Complex number plane"),
    ManimClassNode(name="ParametricFunction", module="manim", is_mobject=True, description="Parametric curve"),
    ManimClassNode(name="ParametricSurface", module="manim", is_mobject=True, description="3D parametric surface"),
    ManimClassNode(name="Sphere", module="manim", is_mobject=True, description="3D sphere"),
]

KNOWN_ANIMATIONS = [
    AnimationNode(name="Write", description="Draws the mobject stroke"),
    AnimationNode(name="Create", description="Creates the mobject"),
    AnimationNode(name="FadeIn", description="Fades in the mobject"),
    AnimationNode(name="FadeOut", description="Fades out the mobject"),
    AnimationNode(name="Transform", description="Morphs one mobject into another"),
    AnimationNode(name="ReplacementTransform", description="Morphs and replaces mobject"),
    AnimationNode(name="MoveToTarget", description="Moves to target position"),
    AnimationNode(name="Rotate", description="Rotates the mobject"),
    AnimationNode(name="ShowCreation", description="Shows creation animation"),
    AnimationNode(name="DrawBorderThenFill", description="Draws border then fills"),
    AnimationNode(name="GrowFromCenter", description="Grows from center point"),
    AnimationNode(name="ApplyMethod", description="Applies a method as animation"),
    AnimationNode(name="Indicate", description="Indicates/highlights mobject"),
    AnimationNode(name="Uncreate", description="Reverse of Create"),
]
