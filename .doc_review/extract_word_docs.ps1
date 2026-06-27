param(
    [Parameter(Mandatory = $true)]
    [string[]]$InputPaths,
    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory
)

$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

function Clean-WordText([string]$Text) {
    if ($null -eq $Text) { return '' }
    return ($Text -replace "[\r\a]+$", '' -replace "\v", "`n")
}

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0

try {
    $index = 0
    foreach ($path in $InputPaths) {
        $index++
        $doc = $word.Documents.Open($path, $false, $true)
        try {
            $paragraphs = @()
            foreach ($p in $doc.Paragraphs) {
                $range = $p.Range
                $text = Clean-WordText $range.Text
                if ([string]::IsNullOrWhiteSpace($text)) { continue }
                $style = ''
                try { $style = [string]$range.Style.NameLocal } catch { $style = [string]$range.Style }
                $listLabel = ''
                try { $listLabel = [string]$range.ListFormat.ListString } catch { }
                $page = $null
                try { $page = [int]$range.Information(1) } catch { }
                $paragraphs += [PSCustomObject]@{
                    index = $paragraphs.Count + 1
                    page = $page
                    style = $style
                    list_label = $listLabel
                    text = $text
                }
            }

            $tables = @()
            $tableIndex = 0
            foreach ($table in $doc.Tables) {
                $tableIndex++
                $rows = @()
                for ($r = 1; $r -le $table.Rows.Count; $r++) {
                    $cells = @()
                    for ($c = 1; $c -le $table.Columns.Count; $c++) {
                        try { $cells += (Clean-WordText $table.Cell($r, $c).Range.Text) }
                        catch { $cells += '[MERGED CELL]' }
                    }
                    $rows += ,$cells
                }
                $page = $null
                try { $page = [int]$table.Range.Information(1) } catch { }
                $tables += [PSCustomObject]@{
                    index = $tableIndex
                    page = $page
                    rows = $table.Rows.Count
                    columns = $table.Columns.Count
                    data = $rows
                }
            }

            $comments = @()
            foreach ($comment in $doc.Comments) {
                $comments += [PSCustomObject]@{
                    author = [string]$comment.Author
                    date = [string]$comment.Date
                    scope = Clean-WordText $comment.Scope.Text
                    text = Clean-WordText $comment.Range.Text
                }
            }

            $revisions = @()
            foreach ($revision in $doc.Revisions) {
                $revisions += [PSCustomObject]@{
                    author = [string]$revision.Author
                    date = [string]$revision.Date
                    type = [int]$revision.Type
                    text = Clean-WordText $revision.Range.Text
                }
            }

            $result = [PSCustomObject]@{
                source = $path
                title = [System.IO.Path]::GetFileNameWithoutExtension($path)
                pages = [int]$doc.ComputeStatistics(2)
                words = [int]$doc.ComputeStatistics(0)
                characters = [int]$doc.ComputeStatistics(3)
                paragraphs_count = $paragraphs.Count
                tables_count = $doc.Tables.Count
                inline_shapes_count = $doc.InlineShapes.Count
                floating_shapes_count = $doc.Shapes.Count
                comments_count = $doc.Comments.Count
                revisions_count = $doc.Revisions.Count
                footnotes_count = $doc.Footnotes.Count
                endnotes_count = $doc.Endnotes.Count
                hyperlinks_count = $doc.Hyperlinks.Count
                sections_count = $doc.Sections.Count
                paragraphs = $paragraphs
                tables = $tables
                comments = $comments
                revisions = $revisions
            }

            $stem = ('{0:D2}_{1}' -f $index, ([System.IO.Path]::GetFileNameWithoutExtension($path) -replace '[^\p{L}\p{Nd}._-]+', '_'))
            $jsonPath = Join-Path $OutputDirectory ($stem + '.json')
            $textPath = Join-Path $OutputDirectory ($stem + '.txt')
            $result | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding UTF8

            $lines = [System.Collections.Generic.List[string]]::new()
            $lines.Add("TITLE: $($result.title)")
            $lines.Add("PAGES: $($result.pages); WORDS: $($result.words); TABLES: $($result.tables_count); IMAGES/SHAPES: $($result.inline_shapes_count + $result.floating_shapes_count)")
            $lines.Add('')
            $lines.Add('=== PARAGRAPHS ===')
            foreach ($p in $paragraphs) {
                $prefix = "[p.$($p.page) | $($p.style)]"
                if ($p.list_label) { $prefix += " [$($p.list_label)]" }
                $lines.Add("$prefix $($p.text)")
            }
            $lines.Add('')
            $lines.Add('=== TABLES ===')
            foreach ($table in $tables) {
                $lines.Add("TABLE $($table.index) (page $($table.page), $($table.rows)x$($table.columns))")
                foreach ($row in $table.data) { $lines.Add(($row -join ' | ')) }
                $lines.Add('')
            }
            if ($comments.Count -gt 0) {
                $lines.Add('=== COMMENTS ===')
                foreach ($comment in $comments) { $lines.Add("[$($comment.author)] $($comment.scope) => $($comment.text)") }
            }
            if ($revisions.Count -gt 0) {
                $lines.Add('=== REVISIONS ===')
                foreach ($revision in $revisions) { $lines.Add("[$($revision.author), type $($revision.type)] $($revision.text)") }
            }
            $lines | Set-Content -LiteralPath $textPath -Encoding UTF8
        }
        finally {
            $doc.Close($false)
            [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($doc)
        }
    }
}
finally {
    $word.Quit()
    [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($word)
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
