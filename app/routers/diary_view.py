# routers/diary_view.py
from fastapi import APIRouter, HTTPException, Depends, Query
from datetime import datetime
from app.schemas.chat import (
    WeeklyDiariesResponse,
    WeeklyDiariesData,
    DiaryEntry,
    DiaryByDateResponse
)
from app.services.diary_service import get_diary_service, DiaryService
from app.routers.auth import get_current_user_id
import traceback

router = APIRouter(prefix="/api/diaries", tags=["Diaries"])


@router.get("/weekly", response_model=WeeklyDiariesResponse, summary="일주일치 일기 조회")
async def get_weekly_diaries(
    days: int = Query(
        default=7,
        ge=1,
        le=30,
        description="조회할 일수 (1일 이상 30일 이하, 기본값: 7일)"
    ),
    user_id: str = Depends(get_current_user_id),
    diary_service: DiaryService = Depends(get_diary_service)
):
    """
    일주일치 일기 조회 (최근 N일)

    **파라미터:**
    - days: 조회할 일수 (기본 7일, 최대 30일)

    **응답:**
    - diaries: 일기 리스트 (날짜순 정렬, 최신순)
    - count: 일기 개수
    """
    try:
        # 일수 제한 (최대 30일)
        if days < 1 or days > 30:
            raise HTTPException(status_code=400, detail="일수는 1일 이상 30일 이하여야 합니다.")

        # 일주일치 일기 조회
        diaries_dict = diary_service.get_weekly_diaries(
            user_id=user_id,
            days=days
        )

        # 딕셔너리 리스트를 DiaryEntry 객체 리스트로 변환
        diaries = [
            DiaryEntry(
                content=diary["content"],
                metadata=diary["metadata"],
                diary_date=diary["diary_date"]  
            )
            for diary in diaries_dict
        ]

        return WeeklyDiariesResponse(
            success=True,
            message=f"최근 {days}일치 일기를 조회했습니다.",
            data=WeeklyDiariesData(
                diaries=diaries,
                count=len(diaries),
                days=days
            )
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"일기 조회 실패: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"일기 조회 실패: {str(e)}")


@router.get("/date/{date}", response_model=DiaryByDateResponse, summary="특정 날짜 일기 조회")
async def get_diary_by_date(
    date: str,
    user_id: str = Depends(get_current_user_id),
    diary_service: DiaryService = Depends(get_diary_service)
):
    """
    특정 날짜의 일기 조회

    하루에 일기는 하나입니다.

    **파라미터:**
    - date: 조회할 날짜 (YYYY-MM-DD 형식, 예: 2025-11-12)

    **응답:**
    - data: 일기 데이터 (없으면 null)
    """
    try:
        # 날짜 형식 검증
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="날짜 형식이 올바르지 않습니다. YYYY-MM-DD 형식으로 입력해주세요. (예: 2025-11-12)"
            )

        # 특정 날짜의 일기 조회
        diary_dict = diary_service.get_diary_by_date(
            user_id=user_id,
            date=date
        )

        if not diary_dict:
            return DiaryByDateResponse(
                success=True,
                message=f"{date} 날짜의 일기를 찾을 수 없습니다.",
                data=None
            )

        # DiaryEntry 객체로 변환
        diary = DiaryEntry(
            content=diary_dict["content"],
            metadata=diary_dict["metadata"],
            diary_date=diary_dict["diary_date"]  
        )

        return DiaryByDateResponse(
            success=True,
            message=f"{date} 날짜의 일기를 조회했습니다.",
            data=diary
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"특정 날짜 일기 조회 실패: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"일기 조회 실패: {str(e)}")
