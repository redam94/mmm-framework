import { useState, useCallback } from 'react';
import { Card, Title, Text, Table, TableHead, TableRow, TableHeaderCell, TableBody, TableCell, Badge, Button, Tab, TabGroup, TabList, TabPanels, TabPanel } from '@tremor/react';
import { CloudArrowUpIcon, TrashIcon, EyeIcon } from '@heroicons/react/24/outline';
import { useDatasets, useUploadData, useDeleteDataset, useDatasetVariables } from '../../api/hooks';
import { useProjectStore } from '../../stores/projectStore';
import { LoadingSpinner, LoadingPage } from '../../components/common/LoadingSpinner';
import type { DatasetInfo } from '../../api/types';

// File drop zone component
function UploadDropzone({ onUpload }: { onUpload: (file: File) => void }) {
  const [isDragging, setIsDragging] = useState(false);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDragIn = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragOut = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);

      const files = e.dataTransfer.files;
      if (files && files.length > 0) {
        onUpload(files[0]);
      }
    },
    [onUpload]
  );

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files;
      if (files && files.length > 0) {
        onUpload(files[0]);
      }
    },
    [onUpload]
  );

  return (
    <div
      className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
        isDragging ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'
      }`}
      onDrag={handleDrag}
      onDragEnter={handleDragIn}
      onDragLeave={handleDragOut}
      onDragOver={handleDrag}
      onDrop={handleDrop}
    >
      <CloudArrowUpIcon className="mx-auto h-12 w-12 text-gray-400" />
      <div className="mt-4">
        <label
          htmlFor="file-upload"
          className="cursor-pointer rounded-md font-medium text-blue-600 hover:text-blue-500"
        >
          <span>Upload a file</span>
          <input
            id="file-upload"
            name="file-upload"
            type="file"
            className="sr-only"
            accept=".csv,.parquet,.xlsx,.xls"
            onChange={handleFileSelect}
          />
        </label>
        <span className="text-gray-500"> or drag and drop</span>
      </div>
      <Text className="text-xs text-gray-500 mt-2">
        CSV, Parquet, or Excel files in MFF format
      </Text>
    </div>
  );
}

// Dataset card component
function DatasetCard({
  dataset,
  isSelected,
  onSelect,
  onDelete,
  onView,
}: {
  dataset: DatasetInfo;
  isSelected: boolean;
  onSelect: () => void;
  onDelete: () => void;
  onView: () => void;
}) {
  return (
    <div
      className={`p-4 rounded-lg border-2 cursor-pointer transition-colors ${
        isSelected ? 'border-blue-500 bg-blue-50' : 'border-gray-200 hover:border-gray-300'
      }`}
      onClick={onSelect}
    >
      <div className="flex justify-between items-start">
        <div>
          <Text className="font-medium">{dataset.filename}</Text>
          <Text className="text-xs text-gray-500">
            {dataset.rows.toLocaleString()} rows × {dataset.columns} columns
          </Text>
        </div>
        <Badge color={dataset.format === 'csv' ? 'blue' : dataset.format === 'parquet' ? 'green' : 'gray'}>
          {dataset.format?.toUpperCase() || dataset.filename.split('.').pop()?.toUpperCase() || 'DATA'}
        </Badge>
      </div>

      <div className="mt-3 flex items-center justify-between">
        <Text className="text-xs text-gray-500">
          {dataset.variables.length} variables
        </Text>
        <div className="flex gap-2">
          <Button
            size="xs"
            variant="secondary"
            icon={EyeIcon}
            onClick={(e) => {
              e.stopPropagation();
              onView();
            }}
          >
            View
          </Button>
          <Button
            size="xs"
            variant="secondary"
            color="red"
            icon={TrashIcon}
            onClick={(e) => {
              e.stopPropagation();
              onDelete();
            }}
          >
            Delete
          </Button>
        </div>
      </div>
    </div>
  );
}

// Variable explorer component
function VariableExplorer({ dataId }: { dataId: string }) {
  const { data: variables, isLoading } = useDatasetVariables(dataId);

  if (isLoading) {
    return <LoadingSpinner size="sm" />;
  }

  if (!variables || variables.length === 0) {
    return <Text className="text-gray-500">No variable statistics available</Text>;
  }

  return (
    <Table>
      <TableHead>
        <TableRow>
          <TableHeaderCell>Variable</TableHeaderCell>
          <TableHeaderCell>Count</TableHeaderCell>
          <TableHeaderCell>Mean</TableHeaderCell>
          <TableHeaderCell>Std</TableHeaderCell>
          <TableHeaderCell>Min</TableHeaderCell>
          <TableHeaderCell>Max</TableHeaderCell>
          <TableHeaderCell>Missing</TableHeaderCell>
        </TableRow>
      </TableHead>
      <TableBody>
        {variables.map((v) => (
          <TableRow key={v.name}>
            <TableCell>{v.name}</TableCell>
            <TableCell>{v.count.toLocaleString()}</TableCell>
            <TableCell>{v.mean.toFixed(2)}</TableCell>
            <TableCell>{v.std.toFixed(2)}</TableCell>
            <TableCell>{v.min.toFixed(2)}</TableCell>
            <TableCell>{v.max.toFixed(2)}</TableCell>
            <TableCell>
              {v.missing > 0 ? (
                <Badge color="red">{v.missing}</Badge>
              ) : (
                <Badge color="green">0</Badge>
              )}
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

export function DataUploadPage() {
  const { data, isLoading } = useDatasets();
  const uploadMutation = useUploadData();
  const deleteMutation = useDeleteDataset();
  const { selectedDataId, setSelectedData } = useProjectStore();
  const [, setViewingDataId] = useState<string | null>(null);

  const handleUpload = useCallback(
    (file: File) => {
      uploadMutation.mutate(file, {
        onSuccess: (data) => {
          setSelectedData(data.data_id);
        },
      });
    },
    [uploadMutation, setSelectedData]
  );

  const handleDelete = useCallback(
    (dataId: string) => {
      if (confirm('Are you sure you want to delete this dataset?')) {
        deleteMutation.mutate(dataId, {
          onSuccess: () => {
            if (selectedDataId === dataId) {
              setSelectedData(null);
            }
          },
        });
      }
    },
    [deleteMutation, selectedDataId, setSelectedData]
  );

  if (isLoading) {
    return <LoadingPage message="Loading datasets..." />;
  }

  const datasets = data?.datasets || [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <Title>Data Management</Title>
        <Text>Upload and explore your MFF format data</Text>
      </div>

      <TabGroup>
        <TabList>
          <Tab>Upload</Tab>
          <Tab>Datasets ({datasets.length})</Tab>
          <Tab>Explore</Tab>
        </TabList>
        <TabPanels>
          {/* Upload tab */}
          <TabPanel>
            <Card className="mt-4">
              <Title className="text-sm">Upload Data File</Title>
              <div className="mt-4">
                <UploadDropzone onUpload={handleUpload} />
              </div>

              {uploadMutation.isPending && (
                <div className="mt-4 flex items-center gap-2">
                  <LoadingSpinner size="sm" />
                  <Text>Uploading...</Text>
                </div>
              )}

              {uploadMutation.isError && (
                <div className="mt-4 p-3 bg-red-50 rounded-lg">
                  <Text className="text-red-600">
                    Upload failed: {(uploadMutation.error as Error).message}
                  </Text>
                </div>
              )}

              {uploadMutation.isSuccess && (
                <div className="mt-4 p-3 bg-green-50 rounded-lg">
                  <Text className="text-green-600">
                    Successfully uploaded {uploadMutation.data.filename}
                  </Text>
                </div>
              )}

              {/* MFF format guide */}
              <div className="mt-6 p-4 bg-gray-50 rounded-lg">
                <Title className="text-sm">MFF Format Requirements</Title>
                <Text className="mt-2 text-sm">
                  Your data file should have these 8 columns:
                </Text>
                <ul className="mt-2 text-sm text-gray-600 list-disc list-inside">
                  <li>Period (date)</li>
                  <li>Geography (optional)</li>
                  <li>Product (optional)</li>
                  <li>Campaign (optional)</li>
                  <li>Outlet (optional)</li>
                  <li>Creative (optional)</li>
                  <li>VariableName</li>
                  <li>VariableValue</li>
                </ul>
              </div>
            </Card>
          </TabPanel>

          {/* Datasets tab */}
          <TabPanel>
            <div className="mt-4 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {datasets.length === 0 ? (
                <Card className="col-span-full">
                  <Text className="text-gray-500 text-center">
                    No datasets uploaded yet. Go to the Upload tab to add data.
                  </Text>
                </Card>
              ) : (
                datasets.map((dataset) => (
                  <DatasetCard
                    key={dataset.data_id}
                    dataset={dataset}
                    isSelected={selectedDataId === dataset.data_id}
                    onSelect={() => setSelectedData(dataset.data_id)}
                    onDelete={() => handleDelete(dataset.data_id)}
                    onView={() => setViewingDataId(dataset.data_id)}
                  />
                ))
              )}
            </div>
          </TabPanel>

          {/* Explore tab */}
          <TabPanel>
            <Card className="mt-4">
              {selectedDataId ? (
                <>
                  <Title className="text-sm">Variable Statistics</Title>
                  <div className="mt-4">
                    <VariableExplorer dataId={selectedDataId} />
                  </div>
                </>
              ) : (
                <Text className="text-gray-500 text-center">
                  Select a dataset from the Datasets tab to explore its variables.
                </Text>
              )}
            </Card>
          </TabPanel>
        </TabPanels>
      </TabGroup>
    </div>
  );
}

export default DataUploadPage;
