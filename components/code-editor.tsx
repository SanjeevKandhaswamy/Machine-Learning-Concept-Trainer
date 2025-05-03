"use client"

import { useState } from "react"
import { Loader2 } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

interface CodeEditorProps {
  defaultCode: string
  dataset: string
  expectedOutput?: string
}

export function CodeEditor({ defaultCode, dataset, expectedOutput }: CodeEditorProps) {
  const [code, setCode] = useState(defaultCode)
  const [output, setOutput] = useState("")
  const [isRunning, setIsRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Simulate code execution
  const runCode = () => {
    setIsRunning(true)
    setError(null)
    setOutput("")

    // Simulate a delay for "execution"
    setTimeout(() => {
      try {
        // This is a simplified simulation
        // In a real implementation, we would use a proper code execution service
        if (code.includes("error") || code.includes("throw")) {
          throw new Error("Your code contains an error. Please check and try again.")
        }

        // Simple validation to check if the code contains expected patterns
        const hasImport = code.includes("import") || code.includes("from")
        const hasDataset = code.includes(dataset.split(".")[0]) || code.includes("data")
        const hasFit = code.includes(".fit(") || code.includes("train")
        const hasPrediction = code.includes("predict") || code.includes("score")

        if (!hasImport || !hasDataset || !hasFit || !hasPrediction) {
          setOutput(
            "Your code ran, but it may be incomplete. Make sure you're importing libraries, loading the dataset, and using the model correctly.",
          )
        } else {
          // If code looks good, show expected output
          setOutput(expectedOutput || "Model trained successfully! Accuracy: 0.92")
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "An unknown error occurred")
      } finally {
        setIsRunning(false)
      }
    }, 1500)
  }

  return (
    <div className="border rounded-lg overflow-hidden">
      <Tabs defaultValue="code">
        <div className="border-b px-4">
          <TabsList className="bg-transparent h-12">
            <TabsTrigger value="code" className="data-[state=active]:bg-muted rounded">
              Code
            </TabsTrigger>
            <TabsTrigger value="dataset" className="data-[state=active]:bg-muted rounded">
              Dataset
            </TabsTrigger>
          </TabsList>
        </div>
        <TabsContent value="code" className="m-0">
          <div className="relative">
            <textarea
              value={code}
              onChange={(e) => setCode(e.target.value)}
              className="font-mono text-sm p-4 w-full h-[300px] bg-muted/50 focus:outline-none resize-none"
              spellCheck="false"
            />
            <div className="p-4 border-t flex justify-between items-center">
              <div className="text-sm text-muted-foreground">Write your code and click Run to execute</div>
              <Button onClick={runCode} disabled={isRunning}>
                {isRunning ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Running...
                  </>
                ) : (
                  "Run Code"
                )}
              </Button>
            </div>
          </div>
        </TabsContent>
        <TabsContent value="dataset" className="m-0">
          <div className="p-4 font-mono text-sm bg-muted/50 h-[300px] overflow-auto">
            <pre>{dataset}</pre>
          </div>
        </TabsContent>
      </Tabs>
      <div className="border-t">
        <div className="p-2 bg-muted text-sm font-medium">Output</div>
        <div className="p-4 font-mono text-sm min-h-[100px] max-h-[200px] overflow-auto">
          {error ? (
            <div className="text-red-500">{error}</div>
          ) : output ? (
            <pre>{output}</pre>
          ) : (
            <div className="text-muted-foreground">Run your code to see the output here</div>
          )}
        </div>
      </div>
    </div>
  )
}
