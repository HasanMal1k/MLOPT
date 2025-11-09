// API functions for model management

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface TrainedModel {
  id: string;
  model_name: string;
  model_type: string;
  algorithm: string;
  metrics: Record<string, number>;
  model_file_size: number;
  created_at: string;
  description?: string;
  tags?: string[];
  feature_columns: string[];
  target_column?: string;
  training_time_seconds?: number;
  status: string;
}

export interface SaveModelRequest {
  task_id: string;
  model_name: string;
  description?: string;
  tags?: string[];
}

export interface ModelStats {
  total_models: number;
  ready_models: number;
  training_models: number;
  failed_models: number;
  total_storage_mb: number;
  last_model_created?: string;
}

/**
 * Save a trained model
 */
export async function saveModel(
  request: SaveModelRequest,
  userId: string
): Promise<{ success: boolean; model_id?: string; message?: string }> {
  try {
    const response = await fetch(`${API_BASE}/models/save?user_id=${userId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to save model');
    }

    return await response.json();
  } catch (error) {
    console.error('Error saving model:', error);
    throw error;
  }
}

/**
 * List all saved models for a user
 */
export async function listModels(
  userId: string,
  status?: string
): Promise<{ models: TrainedModel[]; total_count: number }> {
  try {
    const params = new URLSearchParams({ user_id: userId });
    if (status) params.append('status', status);

    const response = await fetch(`${API_BASE}/models/list?${params}`);

    if (!response.ok) {
      throw new Error('Failed to fetch models');
    }

    return await response.json();
  } catch (error) {
    console.error('Error listing models:', error);
    throw error;
  }
}

/**
 * Generate download URL for a model
 */
export async function getModelDownloadUrl(
  modelId: string,
  userId: string
): Promise<{ download_url: string; model_name: string; file_size: number; expires_in: number }> {
  try {
    const response = await fetch(
      `${API_BASE}/models/download/${modelId}?user_id=${userId}`
    );

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to generate download URL');
    }

    return await response.json();
  } catch (error) {
    console.error('Error getting download URL:', error);
    throw error;
  }
}

/**
 * Delete a model
 */
export async function deleteModel(
  modelId: string,
  userId: string
): Promise<{ success: boolean; message: string }> {
  try {
    const response = await fetch(
      `${API_BASE}/models/delete/${modelId}?user_id=${userId}`,
      {
        method: 'DELETE',
      }
    );

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to delete model');
    }

    return await response.json();
  } catch (error) {
    console.error('Error deleting model:', error);
    throw error;
  }
}

/**
 * Get user model statistics
 */
export async function getModelStats(userId: string): Promise<ModelStats> {
  try {
    const response = await fetch(`${API_BASE}/models/stats?user_id=${userId}`);

    if (!response.ok) {
      throw new Error('Failed to fetch model stats');
    }

    const data = await response.json();
    return data.stats;
  } catch (error) {
    console.error('Error fetching model stats:', error);
    throw error;
  }
}

/**
 * Download a model file
 */
export async function downloadModelFile(
  modelId: string,
  userId: string,
  modelName: string
): Promise<void> {
  try {
    // Get signed URL
    const { download_url } = await getModelDownloadUrl(modelId, userId);

    // Create temporary link and trigger download
    const link = document.createElement('a');
    link.href = download_url;
    link.download = `${modelName}.pkl`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  } catch (error) {
    console.error('Error downloading model:', error);
    throw error;
  }
}
