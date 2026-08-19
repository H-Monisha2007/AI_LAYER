import pytest

@pytest.mark.asyncio
async def test_health_endpoint(async_client):
    response = await async_client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["ok", "degraded"]
    assert data["app_name"] == "DeepForensics"
    assert "database_connected" in data
    assert "cuda_available" in data
