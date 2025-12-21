# ============================================
# API FASTAPI - SISTEMA DE RECOMENDACIÓN DE LIBROS
# ============================================
# 
# Este servidor FastAPI replica la funcionalidad del dashboard Gradio
# permitiendo consumir el sistema de recomendación desde múltiples clientes
# (web, mobile, desktop, etc.)
#
# Endpoints disponibles:
# - GET  /                          - Servir página principal HTML
# - GET  /api/categories            - Obtener lista de categorías
# - GET  /api/tones                 - Obtener lista de tonos emocionales
# - POST /api/recommendations       - Obtener recomendaciones de libros

# ========================
# IMPORTACIONES
# ========================
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os

# FastAPI y gestión de request/response
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Librerías para procesamiento de documentos
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

# Librerías para embeddings de Google Gemini
from google import genai
from langchain_core.embeddings import Embeddings

# Cargar variables de entorno (.env)
load_dotenv()

# ========================
# INICIALIZAR FASTAPI
# ========================
app = FastAPI(
    title="Book Recommendation API",
    description="API para obtener recomendaciones de libros usando búsqueda semántica",
    version="1.0.0"
)

# ========================
# PASO 1: Cargar datos
# ========================
print("📚 Cargando dataset de libros...")
books = pd.read_csv("data/books_with_emotions.csv")

# Mejorar calidad de las imágenes de portadas
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"

# Reemplazar URLs rotas con imagen por defecto
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)
print(f"✓ Dataset cargado: {len(books)} libros")

# ========================
# PASO 2: Configurar cliente de Google Gemini
# ========================
print("🔑 Inicializando cliente Google Gemini...")
try:
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    print("✓ Cliente Gemini inicializado")
except Exception as e:
    print(f"✗ Error inicializando Gemini: {e}")
    raise

# ========================
# PASO 3: Clase personalizada para embeddings
# ========================
class GoogleGeminiEmbeddings(Embeddings):
    """
    Clase que implementa embeddings vectoriales usando Google Gemini.
    
    Permite convertir texto a vectores numéricos para búsqueda semántica.
    Soporta procesamiento en lotes (batches) para eficiencia.
    """
    
    def __init__(
        self,
        model: str = "text-embedding-004",
        batch_size: int = 100,
    ):
        """
        Inicializar el generador de embeddings.
        
        Args:
            model: Modelo a usar (text-embedding-004)
            batch_size: Cuántos textos procesar por request (máx 100)
        """
        self.model = model
        self.batch_size = batch_size

    def embed_query(self, text: str):
        """
        Generar embedding para una SOLA consulta.
        
        Args:
            text: Texto de la consulta del usuario
            
        Returns:
            Vector de números flotantes (embedding)
        """
        result = client.models.embed_content(
            model=self.model,
            contents=text,
        )
        return result.embeddings[0].values

    def embed_documents(self, texts: list[str]):
        """
        Generar embeddings para MÚLTIPLES documentos.
        
        Procesa en lotes para optimizar llamadas a la API.
        
        Args:
            texts: Lista de textos a convertir
            
        Returns:
            Lista de vectores embeddings
        """
        vectors: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            result = client.models.embed_content(
                model=self.model,
                contents=batch,
            )

            for emb in result.embeddings:
                vectors.append(emb.values)

        return vectors

# ========================
# PASO 4: Inicializar base de datos vectorial
# ========================
print("🔍 Inicializando base de datos vectorial Chroma...")
try:
    raw_documents = TextLoader("data/tagged_description.txt", encoding="utf-8").load()
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)
    embeddings = GoogleGeminiEmbeddings(model="text-embedding-004")

    db_books = Chroma.from_documents(
        documents,
        embedding=embeddings,
    )
    print(f"✓ Base de datos vectorial inicializada: {len(documents)} documentos")
except Exception as e:
    print(f"✗ Error inicializando Chroma: {e}")
    raise

# ========================
# PASO 5: Modelos Pydantic para Request/Response
# ========================
class RecommendationRequest(BaseModel):
    """
    Modelo para request de recomendaciones.
    
    Atributos:
        query: Descripción en lenguaje natural del libro deseado
        category: Categoría de libro (default: "All")
        tone: Tono emocional (default: "All")
    """
    query: str
    category: str = "All"
    tone: str = "All"


class BookRecommendation(BaseModel):
    """
    Modelo para una recomendación de libro.
    
    Atributos:
        isbn13: ID único del libro
        title: Título del libro
        authors: Autores
        description: Descripción completa
        truncated_description: Descripción truncada a 30 palabras
        image_url: URL de portada
        category: Categoría del libro
        emotions: Dict con puntuaciones de emociones
    """
    isbn13: int
    title: str
    authors: str
    description: str
    truncated_description: str
    image_url: str
    category: str
    emotions: dict


class RecommendationResponse(BaseModel):
    """
    Modelo para respuesta de recomendaciones.
    
    Atributos:
        query: Query procesada
        category: Categoría filtrada
        tone: Tono aplicado
        total_results: Número de libros recomendados
        books: Lista de libros recomendados
    """
    query: str
    category: str
    tone: str
    total_results: int
    books: list[BookRecommendation]

# ========================
# PASO 6: Función de recomendación semántica
# ========================
def retrieve_semantic_recommendations(
        query: str,
        category: str = "All",
        tone: str = "All",
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:
    """
    Obtener recomendaciones de libros usando búsqueda semántica.
    
    Proceso:
    1. Buscar 50 libros más similares a la consulta (búsqueda vectorial)
    2. Filtrar por categoría si es especificada
    3. Ordenar por emoción si es especificada
    4. Retornar los top 16 libros
    
    Args:
        query: Descripción en lenguaje natural del libro deseado
        category: Categoría de libro ("All", "Fiction", "Nonfiction", etc.)
        tone: Tono emocional ("All", "Happy", "Angry", "Suspenseful", etc.)
        initial_top_k: Número inicial de resultados semánticos (50)
        final_top_k: Número final de recomendaciones (16)
        
    Returns:
        DataFrame con libros recomendados
    """
    
    # Paso 1: Búsqueda semántica
    recs = db_books.similarity_search(query, k=initial_top_k)
    
    # Extraer ISBNs de los resultados
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    
    # Obtener información completa
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    # Paso 2: Filtrar por categoría
    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs = book_recs.sort_values(by="joy", ascending=False)
    elif tone == "Surprising":
        book_recs = book_recs.sort_values(by="surprise", ascending=False)
    elif tone == "Angry":
        book_recs = book_recs.sort_values(by="anger", ascending=False)
    elif tone == "Suspenseful":
        book_recs = book_recs.sort_values(by="fear", ascending=False)
    elif tone == "Sad":
        book_recs = book_recs.sort_values(by="sadness", ascending=False)

    return book_recs


# ========================
# PASO 7: Endpoints de la API
# ========================

@app.get("/")
async def serve_frontend():
    """
    Servir la página HTML del frontend.
    
    Returns:
        Archivo HTML
    """
    return FileResponse("frontend/index.html", media_type="text/html")


@app.get("/api/categories")
async def get_categories():
    """
    Obtener lista de categorías disponibles.
    
    Returns:
        JSON con lista de categorías
    """
    categories = ["All"] + sorted(books["simple_categories"].unique().tolist())
    return {"categories": categories}


@app.get("/api/tones")
async def get_tones():
    """
    Obtener lista de tonos emocionales disponibles.
    
    Returns:
        JSON con lista de tonos
    """
    tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]
    return {"tones": tones}


@app.post("/api/recommendations")
async def get_recommendations(request: RecommendationRequest):
    """
    Obtener recomendaciones de libros.
    
    Args:
        request: RecommendationRequest con query, category y tone
        
    Returns:
        JSON con recomendaciones de libros
        
    Raises:
        HTTPException: Si la query está vacía o hay error en procesamiento
    """
    # Validación
    if not request.query or len(request.query.strip()) == 0:
        raise HTTPException(status_code=400, detail="Query no puede estar vacía")

    try:
        # Obtener recomendaciones
        recommendations = retrieve_semantic_recommendations(
            query=request.query,
            category=request.category,
            tone=request.tone
        )

        # Procesar resultados
        books_response = []
        for _, row in recommendations.iterrows():
            # Truncar descripción
            description = row["description"]
            truncated_desc_split = description.split()
            truncated_description = " ".join(truncated_desc_split[:30]) + "..."

            # Formatear autores
            authors_split = row["authors"].split(";")
            if len(authors_split) == 2:
                authors_str = f"{authors_split[0]} and {authors_split[1]}"
            elif len(authors_split) > 2:
                authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
            else:
                authors_str = row["authors"]

            # Crear objeto de recomendación
            book_obj = BookRecommendation(
                isbn13=int(row["isbn13"]),
                title=row["title"],
                authors=authors_str,
                description=description,
                truncated_description=truncated_description,
                image_url=row["large_thumbnail"],
                category=row["simple_categories"],
                emotions={
                    "anger": float(row["anger"]),
                    "disgust": float(row["disgust"]),
                    "fear": float(row["fear"]),
                    "joy": float(row["joy"]),
                    "sadness": float(row["sadness"]),
                    "surprise": float(row["surprise"]),
                    "neutral": float(row["neutral"]),
                }
            )
            books_response.append(book_obj)

        # Respuesta estructurada
        return RecommendationResponse(
            query=request.query,
            category=request.category,
            tone=request.tone,
            total_results=len(books_response),
            books=books_response
        )

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error procesando recomendaciones: {str(e)}")


# ========================
# PASO 8: Configuración de inicio
# ========================
if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*50)
    print("🚀 Iniciando API FastAPI")
    print("="*50)
    print("\n📍 Servidor en: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("📋 ReDoc: http://localhost:8000/redoc")
    print("\n" + "="*50 + "\n")
    
    # Iniciar servidor Uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
