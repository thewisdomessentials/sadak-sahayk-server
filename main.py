import json
import logging
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Depends, File, Form, Query, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy.orm import selectinload

try:
    from .auth import AuthenticatedUser, verify_bearer_token
    from .audio import transcribe_audio_file
    from .database import (
        Case,
        CaseImage,
        ChatMessage,
        ChatSession,
        OfficerDevice,
        UserLocation,
        ensure_cases_user_name_column,
        ensure_cases_vehicle_columns,
        ensure_chat_message_image_url_column,
        ensure_case_images_image_url_column_type,
        ensure_chat_schema,
        ensure_chat_session_branch_columns,
        ensure_officer_devices_table,
        ensure_user_location_table,
        get_db_session,
    )
    from .models import BranchChatRequest, ChatRequest, DeviceRegisterRequest
    from .rag import ( 
        retrieve_context,
        generate_response,
        generate_structured_response,
        stream_response,
        answer_vision_query,
    )
    from .storage import upload_case_image, upload_chat_message_image
except ImportError:
    from auth import AuthenticatedUser, verify_bearer_token
    from audio import transcribe_audio_file
    from database import (
        Case,
        CaseImage,
        ChatMessage,
        ChatSession,
        OfficerDevice,
        UserLocation,
        ensure_cases_user_name_column,
        ensure_cases_vehicle_columns,
        ensure_chat_message_image_url_column,
        ensure_case_images_image_url_column_type,
        ensure_chat_schema,
        ensure_chat_session_branch_columns,
        ensure_officer_devices_table,
        ensure_user_location_table,
        get_db_session,
    )
    from models import BranchChatRequest, ChatRequest, DeviceRegisterRequest
    from rag import (
        retrieve_context,
        generate_response,
        generate_structured_response,
        stream_response,
        answer_vision_query,
    )
    from storage import upload_case_image, upload_chat_message_image

app = FastAPI(title="SadakSahayak RAG API")
logger = logging.getLogger(__name__)


@app.on_event("startup")
async def startup_event():
    """Initialize database schema and ensure all columns exist"""
    db_session = next(get_db_session())
    try:
        ensure_chat_schema()
        ensure_cases_user_name_column(db_session)
        ensure_cases_vehicle_columns(db_session)
        ensure_chat_message_image_url_column(db_session)
        ensure_case_images_image_url_column_type(db_session)
        ensure_officer_devices_table(db_session)
        ensure_user_location_table(db_session)
        ensure_chat_session_branch_columns(db_session)
        logger.info("Database schema initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database schema: {e}")
    finally:
        db_session.close()


# --- Simple Auth ---
verify_token = verify_bearer_token


def normalize_case_timestamp(value: str) -> str:
    timestamp_value = (value or "").strip()
    if not timestamp_value:
        return timestamp_value

    try:
        epoch_value = float(timestamp_value)
    except ValueError:
        return timestamp_value

    if epoch_value > 1_000_000_000_000:
        epoch_value = epoch_value / 1000

    return datetime.fromtimestamp(epoch_value, tz=timezone.utc).isoformat()


def serialize_case(case: Case) -> dict:
    return {
        "id": case.id,
        "user_id": case.user_id,
        "user_name": case.user_name,
        "reason": case.reason,
        "notes": case.notes,
        "vehicle_number": case.vehicle_number,
        "vehicle_category": case.vehicle_category,
        "latitude": case.latitude,
        "longitude": case.longitude,
        "timestamp": case.timestamp,
        "chat_history": case.chat_history,
        "language": case.language,
        "created_at": case.created_at.isoformat() if case.created_at else None,
        "images": [
            {
                "id": image.id,
                "url": image.image_url,
                "original_filename": image.original_filename,
                "content_type": image.content_type,
                "size_bytes": image.size_bytes,
                "sha256": image.sha256,
            }
            for image in case.images
        ],
    }


def get_or_create_chat_session(db: Session, user: AuthenticatedUser) -> ChatSession:
    """Get or create the global (main) chat session for a user."""
    user_name = user.get("name") or user.get("preferred_username") or user.get("email")
    session = (
        db.query(ChatSession)
        .filter(ChatSession.user_id == user.user_id, ChatSession.session_type == "global")
        .first()
    )
    if session is not None:
        return session

    session = ChatSession(user_id=user.user_id, user_name=user_name, session_type="global")
    db.add(session)
    db.flush()
    return session


def get_session_by_id(db: Session, session_id: int, user_id: str) -> ChatSession | None:
    """Fetch any chat session (global or branch) by id, ensuring it belongs to the user."""
    return (
        db.query(ChatSession)
        .filter(ChatSession.id == session_id, ChatSession.user_id == user_id)
        .first()
    )


def serialize_chat_message(message: ChatMessage) -> dict:
    return {
        "id": message.id,
        "role": message.role,
        "text": message.text,
        "message_type": message.message_type,
        "image_url": message.image_url,  # Include image URL for image-type messages
        "language": message.language,
        "created_at": message.created_at.isoformat() if message.created_at else None,
    }


def build_recent_conversation(db: Session, chat_session: ChatSession, user_id: str, limit: int = 6) -> str:
    recent_messages = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.session_id == chat_session.id,
            ChatMessage.user_id == user_id,
            ChatMessage.message_type != "context",  # skip seeded context messages
        )
        .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
        .limit(limit)
        .all()
    )
    recent_messages.reverse()
    return "\n".join(
        f"{'User' if message.role == 'user' else 'Assistant'}: {message.text}"
        for message in recent_messages
        if message.text
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/devices/register")
async def register_device(
    req: DeviceRegisterRequest,
    user: AuthenticatedUser = Depends(verify_bearer_token),
    db: Session = Depends(get_db_session),
):
    try:
        officer_id = user.user_id
        now = datetime.now(timezone.utc)
        device = (
            db.query(OfficerDevice)
            .filter(OfficerDevice.officer_id == officer_id)
            .first()
        )
        if device is None:
            device = OfficerDevice(
                officer_id=officer_id,
                fcm_token=req.fcm_token,
                updated_at=now,
            )
            db.add(device)
        else:
            device.fcm_token = req.fcm_token
            device.updated_at = now

        db.commit()
        return {
            "status": "ok",
            "officer_id": officer_id,
            "updated_at": device.updated_at.isoformat() if device.updated_at else None,
        }
    except Exception as e:
        db.rollback()
        logger.exception("Failed to register device")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat(
    req: ChatRequest,
    token: AuthenticatedUser = Depends(verify_token),
    db: Session = Depends(get_db_session),
):
    try:
        ensure_chat_schema()
        ensure_chat_session_branch_columns(db)
        # Resolve which session to use: branch session if session_id provided, else global
        if req.session_id:
            chat_session = get_session_by_id(db, req.session_id, token.user_id)
            if chat_session is None:
                raise HTTPException(status_code=404, detail="Branch session not found")
        else:
            chat_session = get_or_create_chat_session(db, token)

        # Build conversation history: for branches include seeded context + last N real messages
        recent_conversation = build_recent_conversation(db, chat_session, token.user_id, limit=6)
        if chat_session.session_type == "branch":
            context_msgs = (
                db.query(ChatMessage)
                .filter(
                    ChatMessage.session_id == chat_session.id,
                    ChatMessage.message_type == "context",
                )
                .order_by(ChatMessage.id.asc())
                .all()
            )
            context_text = "\n".join(
                f"{'User' if m.role == 'user' else 'Assistant'}: {m.text}"
                for m in context_msgs if m.text
            )
            if context_text:
                recent_conversation = f"[Branch context]\n{context_text}\n[Recent messages]\n{recent_conversation}"
        now = datetime.now(timezone.utc)
        user_message = ChatMessage(
            session_id=chat_session.id,
            user_id=token.user_id,
            role="user",
            text=req.message,
            message_type="text",
            language=req.language,
        )
        chat_session.updated_at = now
        db.add(user_message)
        db.commit()

        context = retrieve_context(req.message)

        if req.stream:
            def event_stream():
                try:
                    for chunk in stream_response(
                        req.message,
                        context,
                        req.language,
                        conversation_history=recent_conversation,
                    ):
                        yield f"data: {json.dumps({'delta': chunk})}\n\n"

                    yield f"data: {json.dumps({'done': True, 'context_used': len(context)})}\n\n"
                except Exception as exc:
                    yield f"data: {json.dumps({'error': str(exc)})}\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        structured = generate_structured_response(
            req.message,
            context,
            req.language,
            conversation_history=recent_conversation,
        )
        answer = structured.get("answer", "")
        chat_session.updated_at = datetime.now(timezone.utc)
        db.add(
            ChatMessage(
                session_id=chat_session.id,
                user_id=token.user_id,
                role="assistant",
                text=answer,
                message_type="text",
                language=req.language,
            )
        )
        db.commit()

        return {
            "answer": answer,
            "context_used": len(context),
            "needs_followup": structured.get("needs_followup", False),
            "quick_replies": structured.get("quick_replies", []),
            "intent": structured.get("intent", "general_info"),
            "enforcement_agency": structured.get("enforcement_agency", "Unknown"),
            "resolution_authority": structured.get("resolution_authority", "Unknown"),
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat-session")
async def get_chat_session(
    limit: int = Query(default=50, ge=1, le=500),
    session_id: int | None = Query(default=None),
    user: AuthenticatedUser = Depends(verify_bearer_token),
    db: Session = Depends(get_db_session),
):
    try:
        ensure_chat_schema()
        ensure_chat_message_image_url_column(db)  # Ensure image_url column exists
        ensure_chat_session_branch_columns(db)

        if session_id is not None:
            chat_session = get_session_by_id(db, session_id, user.user_id)
            if chat_session is None:
                raise HTTPException(status_code=404, detail="Session not found")
        else:
            chat_session = get_or_create_chat_session(db, user)
            db.commit()

        messages = (
            db.query(ChatMessage)
            .filter(ChatMessage.session_id == chat_session.id, ChatMessage.user_id == user.user_id)
            .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
            .limit(limit)
            .all()
        )
        messages.reverse()

        return {
            "session_id": chat_session.id,
            "session_type": chat_session.session_type,
            "user_name": chat_session.user_name,
            "messages": [serialize_chat_message(message) for message in messages],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to fetch chat session")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/branch-sessions")
async def list_branch_sessions(
    user: AuthenticatedUser = Depends(verify_bearer_token),
    db: Session = Depends(get_db_session),
):
    """List all branch chat sessions for the authenticated user."""
    try:
        ensure_chat_session_branch_columns(db)
        branches = (
            db.query(ChatSession)
            .filter(ChatSession.user_id == user.user_id, ChatSession.session_type == "branch")
            .order_by(ChatSession.updated_at.desc())
            .all()
        )

        result = []
        for branch in branches:
            # Get first context message as preview
            first_msg = (
                db.query(ChatMessage)
                .filter(ChatMessage.session_id == branch.id)
                .order_by(ChatMessage.id.asc())
                .first()
            )
            preview = (first_msg.text[:80] if first_msg and first_msg.text else "") + ("..." if first_msg and len(first_msg.text or "") > 80 else "")
            result.append({
                "id": branch.id,
                "user_name": branch.user_name,
                "created_at": branch.created_at.isoformat() if branch.created_at else None,
                "updated_at": branch.updated_at.isoformat() if branch.updated_at else None,
                "preview": preview,
            })

        return {"branches": result}
    except Exception as e:
        logger.exception("Failed to list branch sessions")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/branch-chat", status_code=201)
async def create_branch_chat(
    req: BranchChatRequest,
    user: AuthenticatedUser = Depends(verify_bearer_token),
    db: Session = Depends(get_db_session),
):
    """Create a new branch chat session pre-seeded with context messages from the parent chat."""
    try:
        ensure_chat_schema()
        ensure_chat_session_branch_columns(db)

        # Look up or create the global session to use as parent
        global_session = get_or_create_chat_session(db, user)
        db.commit()

        # Create the branch session
        branch_session = ChatSession(
            user_id=user.user_id,
            user_name=global_session.user_name or user.get("name") or user.get("preferred_username") or user.get("email"),
            session_type="branch",
            parent_session_id=global_session.id,
        )
        db.add(branch_session)
        db.flush()

        # Seed the branch with selected context messages
        now = datetime.now(timezone.utc)
        for msg in req.context_messages:
            role = msg.get("role", "user")
            text = msg.get("text", "")
            if not text:
                continue
            db.add(ChatMessage(
                session_id=branch_session.id,
                user_id=user.user_id,
                role=role,
                text=text,
                message_type="context",  # marks these as pre-seeded context
                language=req.language,
                created_at=now,
            ))

        branch_session.updated_at = now
        db.commit()

        logger.info(
            f"Branch session {branch_session.id} created for user {user.user_id} "
            f"with {len(req.context_messages)} context messages"
        )
        return {
            "branch_session_id": branch_session.id,
            "message_count": len(req.context_messages),
        }
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        logger.exception("Failed to create branch chat")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str | None = Form(default=None),
    token: AuthenticatedUser = Depends(verify_token),
):
    try:
        transcript_text = transcribe_audio_file(file, language=language)
        return {"text": transcript_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/vision")
async def vision(
    file: UploadFile = File(...),
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    token: AuthenticatedUser = Depends(verify_token),
    db: Session = Depends(get_db_session),
):
    try:
        ensure_chat_schema()
        ensure_chat_message_image_url_column(db)  # Ensure image_url column exists
        
        if not (file.content_type or "").startswith("image/"):
            raise HTTPException(status_code=400, detail="Only image uploads are supported")
 
        chat_session = get_or_create_chat_session(db, token)
        recent_conversation = build_recent_conversation(db, chat_session, token.user_id, limit=3)
        
        # Read the file content
        image_bytes = await file.read()
        
        # Encode the image as a base64 data URL
        import base64
        mime_type = file.content_type or "image/jpeg"
        base64_str = base64.b64encode(image_bytes).decode("utf-8")
        image_url = f"data:{mime_type};base64,{base64_str}"
        logger.info("Image encoded to base64 successfully for ChatMessage storage")
        
        # Create user message with image URL
        user_message = ChatMessage(
            session_id=chat_session.id,
            user_id=token.user_id,
            role="user",
            text=prompt or "[Image attached]",
            message_type="image",
            image_url=image_url,  # Store the image URL
            language=language,
        )
        db.add(user_message)
        chat_session.updated_at = datetime.now(timezone.utc)
        db.commit()
        logger.info(f"User message stored with image_url: {image_url}")
 
        # Process the image with vision
        result = answer_vision_query(
            image_bytes=image_bytes,
            content_type=file.content_type,
            language=language,
            conversation_history=recent_conversation,
            prompt=prompt,
        )
        
        # Create assistant response message
        chat_session = get_or_create_chat_session(db, token)
        chat_session.updated_at = datetime.now(timezone.utc)
        assistant_message = ChatMessage(
            session_id=chat_session.id,
            user_id=token.user_id,
            role="assistant",
            text=result["answer"],
            message_type="text",
            language=language,
        )
        db.add(assistant_message)
        db.commit()

        return {
            "answer": result["answer"],
            "analysis": result["analysis"],
            "context_used": len(result["context"]),
        }
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        logger.exception(f"Vision endpoint error: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cases")
async def get_cases(
    limit: int = Query(default=20, ge=1, le=100),
    user: AuthenticatedUser = Depends(verify_bearer_token),
    db: Session = Depends(get_db_session),
):
    try:
        ensure_cases_vehicle_columns(db)
        cases = (
            db.query(Case)
            .options(selectinload(Case.images))
            .filter(Case.user_id == user.user_id)
            .order_by(Case.created_at.desc(), Case.id.desc())
            .limit(limit)
            .all()
        )

        return {"cases": [serialize_case(case) for case in cases]}
    except Exception as e:
        logger.exception("Failed to fetch cases")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/create-case")
async def create_case(
    reason: str = Form(...),
    notes: str = Form(...),
    latitude: str = Form(...),
    longitude: str = Form(...),
    timestamp: str = Form(...),
    chat_history: str = Form(...),
    language: str = Form(...),
    user_name: str | None = Form(default=None),
    vehicle_number: str | None = Form(default=None),
    vehicle_category: str | None = Form(default=None),
    images: list[UploadFile] = File(default=[]),
    user: AuthenticatedUser = Depends(verify_bearer_token),
    db: Session = Depends(get_db_session),
):
    try:
        ensure_cases_user_name_column(db)
        ensure_cases_vehicle_columns(db)

        case = Case(
            user_id=user.user_id,
            user_name=user_name or user.get("name") or user.get("preferred_username"),
            reason=reason,
            notes=notes,
            vehicle_number=vehicle_number,
            vehicle_category=vehicle_category,
            latitude=latitude,
            longitude=longitude,
            timestamp=normalize_case_timestamp(timestamp),
            chat_history=chat_history,
            language=language,
        )
        db.add(case)
        db.flush()

        for image in images:
            if image.content_type and not image.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="Only image uploads are supported")

            import base64
            from hashlib import sha256
            image_bytes = await image.read()
            digest = sha256(image_bytes).hexdigest()
            size_bytes = len(image_bytes)
            
            mime_type = image.content_type or "image/jpeg"
            base64_str = base64.b64encode(image_bytes).decode("utf-8")
            image_url = f"data:{mime_type};base64,{base64_str}"

            db.add(
                CaseImage(
                    case_id=case.id,
                    image_url=image_url,
                    original_filename=image.filename,
                    content_type=image.content_type,
                    size_bytes=size_bytes,
                    sha256=digest,
                )
            )

        db.commit()

        return {
            "message": "Case created successfully",
            "case_id": case.id,
            "CaseID": case.id,
        }
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        logger.exception("Failed to create case")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/update-case")
async def update_case(
    case_id: int = Form(...),
    reason: str = Form(...),
    notes: str = Form(...),
    latitude: str = Form(...),
    longitude: str = Form(...),
    timestamp: str = Form(...),
    chat_history: str = Form(...),
    language: str = Form(...),
    user_name: str | None = Form(default=None),
    vehicle_number: str | None = Form(default=None),
    vehicle_category: str | None = Form(default=None),
    images: list[UploadFile] = File(default=[]),
    user: AuthenticatedUser = Depends(verify_bearer_token),
    db: Session = Depends(get_db_session),
):
    try:
        ensure_cases_user_name_column(db)
        ensure_cases_vehicle_columns(db)

        case = (
            db.query(Case)
            .options(selectinload(Case.images))
            .filter(Case.id == case_id, Case.user_id == user.user_id)
            .first()
        )
        if case is None:
            raise HTTPException(status_code=404, detail="Case not found")

        case.user_name = user_name or case.user_name or user.get("name") or user.get("preferred_username")
        case.reason = reason
        case.notes = notes
        case.vehicle_number = vehicle_number if vehicle_number is not None else case.vehicle_number
        case.vehicle_category = vehicle_category if vehicle_category is not None else case.vehicle_category
        case.latitude = latitude
        case.longitude = longitude
        case.timestamp = normalize_case_timestamp(timestamp)
        case.chat_history = chat_history
        case.language = language

        for image in images:
            if image.content_type and not image.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="Only image uploads are supported")

            import base64
            from hashlib import sha256
            image_bytes = await image.read()
            digest = sha256(image_bytes).hexdigest()
            size_bytes = len(image_bytes)
            
            mime_type = image.content_type or "image/jpeg"
            base64_str = base64.b64encode(image_bytes).decode("utf-8")
            image_url = f"data:{mime_type};base64,{base64_str}"

            db.add(
                CaseImage(
                    case_id=case.id,
                    image_url=image_url,
                    original_filename=image.filename,
                    content_type=image.content_type,
                    size_bytes=size_bytes,
                    sha256=digest,
                )
            )

        db.commit()
        db.refresh(case)

        return {
            "message": "Case updated successfully",
            "case_id": case.id,
            "CaseID": case.id,
        }
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        logger.exception("Failed to update case")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/location", status_code=201)
async def record_location(
    latitude: float = Form(...),
    longitude: float = Form(...),
    accuracy: float | None = Form(default=None),
    recorded_at: str | None = Form(default=None),  # ISO-8601 or epoch ms; optional
    user: AuthenticatedUser = Depends(verify_bearer_token),
    db: Session = Depends(get_db_session),
):
    """
    Store a GPS ping from the field officer app.
    The dashboard can pull these later to show officer positions on a map.
    """
    try:
        ensure_user_location_table(db)

        # Parse recorded_at – accept ISO-8601 string or epoch-ms integer string
        if recorded_at:
            try:
                ts = float(recorded_at)
                if ts > 1_000_000_000_000:
                    ts /= 1000
                parsed_at = datetime.fromtimestamp(ts, tz=timezone.utc)
            except ValueError:
                parsed_at = datetime.fromisoformat(recorded_at.replace("Z", "+00:00"))
        else:
            parsed_at = datetime.now(timezone.utc)

        loc = UserLocation(
            user_id=user.user_id,
            user_name=getattr(user, "name", None) or getattr(user, "preferred_username", None),
            latitude=latitude,
            longitude=longitude,
            accuracy=accuracy,
            recorded_at=parsed_at,
        )
        db.add(loc)
        db.flush()

        # Keep only the last 3 locations for this user to prevent bloating the database
        user_locs = (
            db.query(UserLocation)
            .filter(UserLocation.user_id == user.user_id)
            .order_by(UserLocation.recorded_at.desc(), UserLocation.id.desc())
            .all()
        )
        if len(user_locs) > 3:
            for old_loc in user_locs[3:]:
                db.delete(old_loc)

        db.commit()
        db.refresh(loc)

        logger.info(
            f"📍 Location ping saved: user={user.user_id} "
            f"lat={latitude} lon={longitude} acc={accuracy}"
        )
        return {
            "message": "Location recorded",
            "id": loc.id,
            "user_id": loc.user_id,
            "latitude": loc.latitude,
            "longitude": loc.longitude,
            "recorded_at": loc.recorded_at.isoformat(),
        }
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        logger.exception("Failed to record location")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
